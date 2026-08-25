from CODE.models import FusionLayer
import numpy as np
import torch
import bisect
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import random
from torch.cuda.amp import autocast, GradScaler
import pickle





class createDS(Dataset):
    def __init__(self, data, device='cpu'):
        super(createDS, self).__init__()
        self.data = data
        self.device = device
        self.num_inc = len(data[7][0][0])
        self.choices = [i for i in range(self.num_inc)]
        self.max_reports = len(data[3][0])
        

    def __len__(self):
        return len(self.data[0])

    def random(self):
        idx = [random.choice(self.choices) for i in range(self.max_reports)]
        return torch.tensor(idx).unsqueeze(-1)

    def __getitem__(self, idx):
        dic = {}
        dic['x'] = self.data[0][idx].to(self.device)
        dic['x_times'] = self.data[1][idx].unsqueeze(-1).to(self.device)
        dic['x_mask'] = self.data[2][idx].to(self.device)
        dic['embedding'] = self.data[3][idx].to(self.device)
        dic['embedding_times'] = self.data[4][idx].unsqueeze(-1).to(self.device)
        dic['embedding_mask'] = self.data[5][idx].to(self.device)
        dic['y'] = self.data[6][idx].to(self.device)
        dic['cls_inc'] = torch.stack([reports[random.choice(self.choices)] \
                                      for reports in self.data[7][idx]])\
                                      .to(self.device)
        return dic



    

    
class Normalizer:
    def __init__(self, data, device):
        # data: list of lists
        mean = np.zeros(len(data[0][0]))
        count = 0
        for i in data:
            for j in i:
                mean += np.array(j)
                count += 1
        mean = mean / count
        std = np.zeros(len(data[0][0]))
        for i in data:
            for j in i:
                std += (np.array(j) - mean) ** 2
        std = (std / count) ** 0.5
        self.mean = torch.tensor(mean.tolist()).to(device)
        self.mean.requires_grad = False
        self.std = torch.tensor(std.tolist()).to(device) + 0.00001
        self.std.requires_grad = False
    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def metric_improved(metrics, best_metrics):
    # metrics: list, best_metrics: list
    improved = sum(np.sign(np.array(metrics) - np.array(best_metrics)))
    if improved > 0:
        return True
    else:
        return False

def loss_improved(loss, best_loss):
    # metrics: list, best_metrics: list
    improved = best_loss - loss
    if improved > 0:
        return True
    else:
        return False
    

@torch.no_grad()
def cal_metrics(model, data, act):
    ### the next line can be adjusted for sentence deletion
    #model.diffusion.highest_percent = 0.3
    model.eval()
    true_y, pred_y, pred_pro = [], [], []
    for batch in data:
        x = batch['x']
        x_times = batch['x_times']
        x_mask = batch['x_mask']
        embedding = batch['embedding']
        embedding_times = batch['embedding_times']
        embedding_mask = batch['embedding_mask']
        y = batch['y'].flatten()
        probs = act(model(x, x_times, x_mask, embedding, embedding_times,
                    embedding_mask, testing=True))[:,1]
        model.diffusion.reset_params_ELBO_0_tp()
        pred_label = torch.round(probs).flatten()
        true_y.extend(y.flatten().tolist())
        pred_y.extend(pred_label.tolist())
        pred_pro.extend(probs.flatten().tolist())
    aucroc = roc_auc_score(true_y, pred_pro)
    aucpr = average_precision_score(true_y, pred_pro)
    f1 = f1_score(true_y, pred_y)
    return f1, aucpr, aucroc



@torch.no_grad()
def cal_pretrain_loss(model, data,args):
    tot_loss = 0.
    for batch in data:
        embedding = batch['embedding']
        cls_inc = batch['cls_inc']
        _, tp_densities = model.diffusion.get_encode_tp(embedding)
        note_time_mask = batch['embedding_mask']
        with autocast():
            loss = model.diffusion.loss(embedding, cls_inc, note_time_mask,
                                        None, tp_densities, cond=None,
                                        pretrain=True)
        tot_loss += loss.item()
        model.diffusion.reset_params_ELBO_0_tp()
    return tot_loss


def pretrain_tp_predictor(model, save_path, data_tr, data_val, 
                          optimizer,args, scaler):

    def validate_tp(model, embedding):
        _, tp = model.diffusion(embedding)

        # t_prime estimator hyper-parameter selection criteria
        # low complete note mean tp
        print('mean tp: ', np.mean(tp))
        
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(0, args.epochs):
        for batch in data_tr:
            embedding = batch['embedding']
            cls_inc = batch['cls_inc']
            note_time_mask = batch['embedding_mask']
            _, tp_densities = model.diffusion.get_encode_tp(embedding)
            with autocast():
                loss = model.diffusion.loss(embedding, cls_inc, note_time_mask,
                                        None, tp_densities, cond=None,
                                        pretrain=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model.diffusion.reset_params_ELBO_0_tp()
        metrics = cal_pretrain_loss(model, data_val, args)
        if loss_improved(metrics,best_loss):
            best_loss = metrics
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            

    model.load_state_dict(torch.load(save_path))
    # freeze tp
    for p in model.diffusion.t_prime_predictor.parameters():
        p.requires_grad = False

    validate_tp(model, embedding)
    return model
        
def train_classifier(model, data_tr, data_val, epochs,
                     optimizer, save_path, scaler, Softmax, tol=12, pretrain=False,
                     accumulate_step=1):
    best_metrics = [0.0, 0.0, 0.0] # f1, aucpr, aucroc
    best_epoch = 0
    for epoch in range(0, epochs):
        model.train()
        for mini_step, batch in enumerate(data_tr):
            x = batch['x']
            x_times = batch['x_times']
            x_mask = batch['x_mask']
            embedding = batch['embedding']
            embedding_times = batch['embedding_times']
            embedding_mask = batch['embedding_mask']
            cls_inc = batch['cls_inc']
            y = batch['y'].flatten()
            with autocast():
                loss = model.loss(x, x_times, x_mask, embedding, embedding_times,
                                  embedding_mask, y, cls_inc,
                              pretrain=pretrain)
            scaler.scale(loss).backward()
            if (mini_step+1) % accumulate_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            model.diffusion.reset_params_ELBO_0_tp()
        metrics = cal_metrics(model, data_val,Softmax)
        if metric_improved(metrics, best_metrics):
            best_metrics = metrics
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch
        # early stop
        if (epoch - best_epoch) > tol:
            break
    model.load_state_dict(torch.load(save_path))
    return model, optimizer, scaler
    

def train_step(data_tr1, data_val, data_te, args, tol=12):
    args.emb_size = len(data_tr1[3][0][0])
    args.time_series_size = len(data_tr1[0][0][0])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # tol: tolerance for early stop
    normalizer = Normalizer(data_tr1[0], args.device)
    Softmax = torch.nn.Softmax(dim=1)
    accumulate_step = args.accumulate_step
    torch.manual_seed(args.seed)
    model = FusionLayer(args.num_modalities,
                                   args.emb_size,
                                   args.time_series_size,
                                   args.hidden_dim,
                                   args.output_dim,
                        args.num_layers,
                        args.num_layers_pred,
                        args.num_experts,
                        args.num_routers,
                        args.top_k,
                        normalizer,
                        args,
                        num_heads = args.num_heads,
                        dropout=args.dropout)
    model.to(args.device)
    scaler = GradScaler()
    
    data_tr = DataLoader(createDS(data_tr1, device=args.device),
                         batch_size=args.batch_size, shuffle=True)
    data_val = DataLoader(createDS(data_val, device=args.device),
                         batch_size=args.batch_size, shuffle=False)
    data_te = DataLoader(createDS(data_te, device=args.device),
                         batch_size=args.batch_size, shuffle=False)
    save_path = args.root + 'saved.pth'
    ### pretrain t' predictor
    if args.use_diffusion:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_learning_rate)
        model = pretrain_tp_predictor(model, save_path, data_tr, data_val, \
                                  optimizer,args, scaler)

        
    ################
        ### pretrain MoE-Retirever
        model, optimizer, scaler = train_classifier(model, data_tr, data_val, 10,
                     optimizer, save_path, scaler, Softmax, tol=12, pretrain=True)
        ############
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if accumulate_step != 1:
        data_tr = DataLoader(createDS(data_tr1, device=args.device),
                         batch_size=int(args.batch_size / accumulate_step),
                             shuffle=True)
    model, optimizer, scaler = train_classifier(model, data_tr, data_val, args.epochs,
                     optimizer, save_path, scaler, Softmax, tol=12,
                                                accumulate_step=accumulate_step)
    
    metrics = cal_metrics(model, data_te, Softmax)
    return metrics
    

def train_CODE(dataset, model_name, bert, tokenizer,
                        args, ratio, temp_data, load_split_data, seed,
                        tol=12):

    test_path = args.root + dataset + '_evaluation.txt'
    file = open(test_path, 'w')
    file.close()
    ## tp predictor params ##
    pretrain_learning_rates = [0.0001,0.00001]
    tp_embed_dims = [160]
    tp_layers = [3]
    lambda1s = [0.5]
    
    ###########
    lrs = [0.00001]
    diff_embed_dims = [128, 160]
    num_layers = [3]
    diff_layers = [2,3]
    lambda0s = [0.1]
    num_experts = [16,32]
    top_ks = [2,4]
    highest_percents = [0.3] # alias: lambda2
    embed_dims = [160]
    
    
    args.ts_cond = True
    args.use_diffusion = True
    args.multimodal = True # [CLS] only if False
    if args.multimodal == False or args.use_diffusion == False:
        args.ts_cond = False
    args.num_modalities = 2 if args.multimodal else 1
    print('is multimodal: ', args.multimodal,
          ', use diffusion: ', args.use_diffusion,
          ', is conditional: ', args.ts_cond)


    
    # tips to choose hyper-params
    # fine-tune t' predictor with lambda1 and lambda2 first then fixed,
    # low tp_incomplete is preferred
    # use best top_ks, num_experts, num_layers, lrs for optimal MoE-Retriever
    for pretrain_learning_rate in pretrain_learning_rates:
        for lr in lrs:
            for diff_embed_dim in diff_embed_dims:
                for diff_layer in diff_layers:
                    for lambda0 in lambda0s:
                        for lambda1 in lambda1s:
                            for highest_percent in highest_percents:
                                for num_expert in num_experts:
                                    for top_k in top_ks:
                                        for num_layer in num_layers:
                                            for tp_embed_dim in tp_embed_dims:
                                                for tp_layer in tp_layers:
                                                    args.tp_embed_dim = tp_embed_dim
                                                    args.tp_layer = tp_layer
                                                    args.pretrain_learning_rate = pretrain_learning_rate
                                                    args.lr = lr
                                                    args.diff_embed_dim = diff_embed_dim
                                                    args.diff_layer = diff_layer
                                                    args.num_layer = num_layer
                                                    args.lambda0 = lambda0
                                                    args.lambda1 = lambda1
                                                    args.highest_percent = highest_percent
                                                    args.num_expert = num_expert
                                                    args.top_k = top_k
                                                    args.embed_dim = embed_dims[0]
                                                    metrics = []
                                                    for rep in range(0, args.replication):
                                                        data_tr, data_val, data_te = load_split_data(dataset,model_name,
                                                        bert, tokenizer,
                                                        ratio=ratio,
                                                             timestamp='hour',
                                                             temp_data=temp_data,
                                                             seed = seed**(rep+1))
                                                        metrics.append(train_step(data_tr, data_val, data_te, args, tol=tol))
                                                    metrics = np.array(metrics)
                                                    f1_mean = np.mean(metrics[:,0])
                                                    f1_std = np.std(metrics[:,0])
                                                    aucpr_mean = np.mean(metrics[:,1])
                                                    aucpr_std = np.std(metrics[:,1])
                                                    aucroc_mean = np.mean(metrics[:,2])
                                                    aucroc_std = np.std(metrics[:,2])
                                                    with open(test_path, 'a') as file:
                                                        params = 'pretrain_learning_rate = ' + str(pretrain_learning_rate) + ' lr = ' + \
                                             str(lr) + ' diff_embed_dim = ' + str(diff_embed_dim) \
                                             + ' tp_embed_dim = ' +str(tp_embed_dim) + ' tp_layer = ' + str(tp_layer)\
                                     + ' diff_layer = ' + str(diff_layer) + ' lambda0 = ' + str(lambda0)+\
                                     ' lambda1 = ' + str(lambda1) + ' highest_percent' + str(highest_percent)\
                                     + ' num_expert = ' + str(num_expert) + ' top_k = ' + str(top_k) + ' num_layer' + str(num_layer)
                                                        file.write(params + '    f1 score -- mean: ' + str(f1_mean) + ', std: ' +\
                                           str(f1_std) +  '  AUCPR -- mean: ' + \
                                   str(aucpr_mean) + ', std: ' + str(aucpr_std) + \
                                   '  AUCROC -- mean: ' + str(aucroc_mean) + \
                                   ', std: ' + str(aucroc_std) + '\n')
        
    
    
