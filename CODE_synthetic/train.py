from CODE_synthetic.models import MULTCrossModel
import numpy as np
import torch
import bisect
from torch.utils.data import Dataset, DataLoader
import random
from torch.cuda.amp import autocast, GradScaler
import warnings
import pickle

warnings.filterwarnings("ignore")




class createDS(Dataset):
    def __init__(self, data, device='cpu'):
        super(createDS, self).__init__()
        self.data = data
        self.device = device
        self.num_inc = len(data[1][0][0])
        self.choices = [i for i in range(self.num_inc)]
        

    def __len__(self):
        return len(self.data[0])

    def random(self):
        idx = [random.choice(self.choices) for i in range(self.max_reports)]
        return torch.tensor(idx).unsqueeze(-1)

    def __getitem__(self, idx):
        dic = {}
        dic['embedding'] = self.data[0][idx].to(self.device)
        dic['cls_inc'] = torch.stack([reports[random.choice(self.choices)] \
                                      for reports in self.data[1][idx]])\
                                      .to(self.device)
        dic['emb_gt'] = self.data[2][idx].to(self.device)
        dic['p_id'] = self.data[3][idx].to(self.device)
        return dic




@torch.no_grad()
def cal_t0(tp, note_time_mask_list):
    mask = note_time_mask_list.float().mean(-1).flatten()
    tp1 = tp[:,:,0].flatten().tolist()
    t0 = extract_valid(tp1, mask)
    print('expect t0', sum(t0)/len(t0))

    

def metric_improved(loss, best_loss):
    # metrics: list, best_metrics: list
    improved = best_loss - loss
    if improved > 0:
        return True
    else:
        return False




def extract_valid(x,mask):
    result = []
    for i,j in zip(x,mask):
        if j == 1:
            result.append(i)
    return result





class Normalizer:
    def __init__(self, data, device='cpu'):
        # data: tensor
        self.max = data.float().max(0)[0].to(device)
        self.max.requires_grad = False
        self.min = data.float().min(0)[0].to(device)
        self.min.requires_grad = False
        self.max_min = self.max - self.min
    def __call__(self, x):
        return x # fix to no normalization
        #length = torch.sum(x ** 2, -1) ** 0.5
        #return x / length.unsqueeze(-1)
        #return ((x - self.min) / self.max_min).float()


def cosine_sim(x,y):
    # x,y: [bs,seq,dim] or [bs, dim]
    dim = x.shape[-1]
    x1=x.view(-1, dim)
    y1=y.view(-1, dim)
    result = torch.sum(x1*y1,-1) / (torch.sqrt(torch.sum(x1 ** 2, -1)) \
                        * torch.sqrt(torch.sum(y1 ** 2, -1)))
    return np.array(result.flatten().tolist())

def cal_Euclidean(x,y):
    # x,y: [bs, seq, 768] or [bs, 768]
    x1 = x.view(-1,768)
    y1 = y.view(-1,768)
    Euclidean = torch.sum((x1-y1)**2,-1).sqrt()
    return Euclidean.flatten().tolist()


    

@torch.no_grad()
def cal_tp(model, data, sentence_dels, normalizer):
    tp_com,tp_inc,t0_com,t0_inc = [],[],[],[]
    naive_complete, naive_incomplete, ours_complete, ours_incomplete = [],[],[],[]
    ours_complete_mask, ours_incomplete_mask = [],[]
    complete_inst, percent_highest = [], []
    naive_complete_cosine, naive_incomplete_cosine, \
    ours_complete_mask_cosine, ours_incomplete_mask_cosine = [],[],[],[]
    
    
    
    for batch in data:
        model.diffusion.reset_params_ELBO_0_tp()
        embedding1 = batch['embedding']
        note_time_mask_list = torch.ones_like(embedding1)
        emb_gt = batch['emb_gt']
        p_id = batch['p_id'].flatten().tolist()
        
        t0,tp = model.diffusion(embedding1)
        infer_emb = model(embedding1,
                note_time_mask_list, train=False)
        

        
        
        emb_gt = normalizer(emb_gt)
        infer_emb = normalizer(infer_emb)
        
        ours_mask_pred = normalizer(get_highest_percent(model, embedding1))
        p_tp_0 = model.diffusion.t_prime_predictor(embedding1)[:,:,0]
        tp_0_mask = model.diffusion.rank_mask(p_tp_0, model.diffusion.lambda2)\
                    .flatten().tolist()
        percent_highest.extend(tp_0_mask)
        embedding1 = normalizer(embedding1)
        for idx,i in enumerate(p_id):
            num_del = sentence_dels[i]

            naive_pred = embedding1[idx]
            ours_pred = infer_emb[idx]
            
            gt = emb_gt[idx]
            naive_error = ((naive_pred - gt) ** 2).flatten().tolist()
            naive_cosine = cosine_sim(naive_pred, gt)
            ours_error = ((ours_pred - gt) ** 2).flatten().tolist()
            
            ours_mask_pred_error = ((ours_mask_pred[idx] - gt) ** 2)\
                                   .flatten().tolist()
            ours_mask_pred_cosine = cosine_sim(ours_mask_pred[idx], gt)
            
            if num_del == 0:
                tp_com.append(tp[idx])
                t0_com.append(t0[idx])
                naive_complete.append(naive_error)
                ours_complete.append(ours_error)
                complete_inst.append(1.0)
                ours_complete_mask.append(ours_mask_pred_error)

                naive_complete_cosine.append(naive_cosine)
                ours_complete_mask_cosine.append(ours_mask_pred_cosine)
                
            else:
                tp_inc.append(tp[idx])
                t0_inc.append(t0[idx])
                naive_incomplete.append(naive_error)
                ours_incomplete.append(ours_error)
                complete_inst.append(0.0)
                ours_incomplete_mask.append(ours_mask_pred_error)

                naive_incomplete_cosine.append(naive_cosine)
                ours_incomplete_mask_cosine.append(ours_mask_pred_cosine)
    t0_com_mean = sum(t0_com) / len(t0_com)
    tp_com_mean = sum(tp_com) / len(tp_com)
    t0_inc_mean = sum(t0_inc) / len(t0_inc)
    tp_inc_mean = sum(tp_inc) / len(tp_inc)
    percent_highest = np.array(percent_highest)
    complete_inst = np.array(complete_inst)
    print('t0 complete ', t0_com_mean, ', t0 incomplete ', t0_inc_mean,
          'tp complete ', tp_com_mean, ', tp incomplete ', tp_inc_mean,
          'naive_complete_err ', np.array(naive_complete).mean(),
          'naive_incomplete_err ', np.array(naive_incomplete).mean(),
          'naive_complete_cosine ', np.array(naive_complete_cosine).mean(),
          'naive_incomplete_cosine ', np.array(naive_incomplete_cosine).mean(),
          'ours_complete_err ', np.array(ours_complete).mean(),
          'ours_incomplete_err ', np.array(ours_incomplete).mean(),
          'ours_complete_high_percent_err ', np.array(ours_complete_mask).mean(),
          'ours_incomplete__high_percent_err ', np.array(ours_incomplete_mask).mean(),
          'ours_complete_mask_cosine ', np.array(ours_complete_mask_cosine).mean(),
          'ours_incomplete_mask_cosine ', np.array(ours_incomplete_mask_cosine).mean(),
          'highest percent true positive: ', sum(percent_highest*complete_inst)/sum(percent_highest)
          )
    

    
                

def get_highest_percent(model, embedding):
    note_time_mask_list = torch.ones_like(embedding)
    tp_densities_txt = model.diffusion.t_prime_predictor(embedding)[:,:,0]
    p_tp_0 = tp_densities_txt[:,0].unsqueeze(-1)
    bs = p_tp_0.shape[0]
    percentile_complete = model.diffusion.lambda2
    tp_0_mask = model.diffusion.rank_mask(p_tp_0, percentile_complete).view(bs,1,1)
    infer_emb = model(embedding,
                note_time_mask_list, train=False)
    infer_emb = tp_0_mask * embedding + (1 - tp_0_mask) * infer_emb
    return infer_emb

    

@torch.no_grad()
def cal_metrics(model, data, normalizer, args, sentence_dels):
    true_y, pred_y, pred_pro,sqrt_error, Cosine_dist = [], [], [], [], []
    percentile_complete = args.lambda1
    p_ids = []
    real_complete_Euclidean, real_incomplete_Euclidean = [],[]
    real_complete_cos, real_incomplete_cos = [],[]
    
    
    for batch in data:
        embedding = batch['embedding']
        note_time_mask_list = torch.ones_like(embedding)
        emb_gt = normalizer(batch['emb_gt'])
        p_id = batch['p_id'].flatten().tolist()
        
        infer_emb = normalizer(get_highest_percent(model, embedding))
        
        sqrt_error.extend(cal_Euclidean(infer_emb, emb_gt))
        Cosine_dist.extend((1 - cosine_sim(infer_emb, emb_gt)).tolist())
        p_ids.extend(p_id)
    for idx, p_id in enumerate(p_ids):
        num_del = sentence_dels[p_id]
        if num_del == 0:
            real_complete_Euclidean.append(sqrt_error[idx])
            real_complete_cos.append(Cosine_dist[idx])
        else:
            real_incomplete_Euclidean.append(sqrt_error[idx])
            real_incomplete_cos.append(Cosine_dist[idx])
    Euclidean_all = np.array(sqrt_error).mean()
    Cos_dist_all = np.array(Cosine_dist).mean()
    Euclidean_complete = np.array(real_complete_Euclidean).mean()
    Cos_dist_complete = np.array(real_complete_cos).mean()
    Euclidean_incomplete = np.array(real_incomplete_Euclidean).mean()
    Cos_dist_incomplete = np.array(real_incomplete_cos).mean()
    return Euclidean_all, Euclidean_complete, Euclidean_incomplete, Cos_dist_all, Cos_dist_complete,\
           Cos_dist_incomplete 


@torch.no_grad()
def cal_loss(model, data):
    tot_loss = 0.
    for batch in data:
        embedding = batch['embedding']
        note_time_mask_list = torch.ones_like(embedding)
        cls_inc = batch['cls_inc']
        with autocast():
            loss = model.UpperBound(embedding,
            note_time_mask_list, cls_inc)
        tot_loss += loss
        model.diffusion.reset_params_ELBO_0_tp()
    return tot_loss


@torch.no_grad()
def cal_pretrain_loss(model, data,args):
    tot_loss = 0.
    for batch in data:
        embedding = batch['embedding']
        note_time_mask_list = torch.ones_like(embedding)
        cls_inc = batch['cls_inc']
        with autocast():
            loss = model(embedding,
                note_time_mask_list,train=True, x_cls_inc=cls_inc,
                             use_checkpoint=args.checkpoint,
                             pretrain=True)
        tot_loss += loss.item()
        model.diffusion.reset_params_ELBO_0_tp()
    return tot_loss
    

    

def train_step(data_tr, data_val, data_te, args, sentence_dels,
               tol=12,pre_tr_epoch=12):
    # data_val[0] = given CLS [bs,1,768] torch.tensor
    # data_val[1] = corrupted CLS [bs,1,3,768]
    # data_val[2] = GT CLS [bs,1,768]
    # data_val[3] = IDs [bs]
    args.dx = 0
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.text_seq_num = 1
    normalizer = Normalizer(data_tr[2].squeeze(1), device=args.device)
    torch.manual_seed(args.seed)
    model = MULTCrossModel(args)
    model.to(args.device)
    scaler = GradScaler()
    optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n and 'diffusion' not in n]},
                {'params':[p for n, p in model.named_parameters() if 'bert' in n], 'lr': \
                 args.txt_learning_rate},
                {'params':[p for n, p in model.named_parameters() if 'diffusion' in n], 'lr': \
                 args.pretrain_learning_rate} 
            ], lr=args.ts_learning_rate)
    data_tr = DataLoader(createDS(data_tr, device=args.device),
                         batch_size=args.batch_size, shuffle=True)
    data_val = DataLoader(createDS(data_val, device=args.device),
                         batch_size=args.batch_size, shuffle=True)
    data_te = DataLoader(createDS(data_te, device=args.device),
                         batch_size=args.batch_size, shuffle=True)
    save_path = args.root + 'saved.pth'


    ### pretrain #####
    best_loss = np.inf
    best_epoch = 0
    for epoch in range(0, args.epochs):
        for batch in data_tr:
            embedding = batch['embedding']
            cls_inc = batch['cls_inc']
            note_time_mask_list = torch.ones_like(embedding)
            with autocast():
                loss = model(embedding,
                note_time_mask_list,train=True, x_cls_inc=cls_inc,
                             use_checkpoint=args.checkpoint,
                             pretrain=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model.diffusion.reset_params_ELBO_0_tp()
        metrics = cal_pretrain_loss(model, data_val, args)
        if metric_improved(metrics,best_loss):
            best_loss = metrics
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            
        # early stop
        if (epoch - best_epoch) > tol:
            break
    model.load_state_dict(torch.load(save_path))
    # freeze tp
    for p in model.diffusion.t_prime_predictor.parameters():
        p.requires_grad = False
    ################
    optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n and 'diffusion' not in n]},
                {'params':[p for n, p in model.named_parameters() if 'bert' in n], 'lr': \
                 args.txt_learning_rate},
                {'params':[p for n, p in model.named_parameters() if 'diffusion' in n], 'lr': \
                 args.diff_learning_rate} 
            ], lr=args.ts_learning_rate)
    best_loss = np.inf # f1, aucpr, aucroc
    best_epoch = 0
    
    for epoch in range(0, args.epochs):
        for batch in data_tr:
            embedding = batch['embedding']
            cls_inc = batch['cls_inc']
            note_time_mask_list = torch.ones_like(embedding)
            with autocast():
                loss = model(embedding,
                note_time_mask_list,train=True, x_cls_inc=cls_inc,
                             use_checkpoint=args.checkpoint)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            model.diffusion.reset_params_ELBO_0_tp()
        metrics = cal_loss(model, data_val)
        if metric_improved(metrics,best_loss):
            best_loss = metrics
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            
        # early stop
        if (epoch - best_epoch) > tol:
            break
    model.load_state_dict(torch.load(save_path))
    metrics = cal_metrics(model, data_te, normalizer, args, sentence_dels)
    #cal_tp(model, data_te, sentence_dels, normalizer)
    return metrics
    
def load_delete_num(dataset_dir, incomplete_proportion):
    incomplete = str(int(100*incomplete_proportion))
    file = open('./'+dataset_dir+f'/incomplete{incomplete}/sent_delete.pickle'+'', 'rb')
    data = pickle.load(file)
    file.close()
    return data # {id:num sent, ...}
    




    
def train_CODE_synthetic(dataset_dir, incomplete_proportion,model_name,
                         bert, tokenizer,
                        args, ratio, temp_data, load_split_data, seed,
                        tol=12):

    test_path = args.root + 'synthetic_evaluation.txt'
    file = open(test_path, 'w')
    file.close()
    ts_learning_rates = [0.001]
    diff_learning_rates = [0.0001]
    ## tp predictor params ##
    pretrain_learning_rates = [0.0001]
    tp_embed_dims = [160]
    tp_layers = [3]
    lambda1s = [0.5]
    ###########
    diff_embed_dims = [128]
    diff_layers = [2]
    embed_dims = [160]
    layers = [3]
    cross_layers = [3]
    num_heads = [8]
    hidden_autos = [512]
    latent_dim_autos = [256]


    lambda0s = [0.5]
    lambda2s = [0.2] # lambda2 = highest_percent
    args.checkpoint = False
    # mortality: ts_learning_rates = [0.001,0.0001,0.00001],
    # diff_learning_rates = [0.00001,0.001] diff_embed_dims = [160],
    # diff_layers = [3]


    ## load num sentences deleted
    sent_delete = load_delete_num(dataset_dir, incomplete_proportion)
    
    for lambda1 in lambda1s:
        ts_learning_rate = ts_learning_rates[0]
        for diff_learning_rate in diff_learning_rates:
            for lambda0 in lambda0s:
                embed_dim = embed_dims[0]
                for layer in layers:
                    for lambda2 in lambda2s:
                        cross_layer = cross_layers[0]
                        num_head = num_heads[0]
                        for diff_layer in diff_layers:
                            hidden_auto = hidden_autos[0]
                            for diff_embed_dim in diff_embed_dims:
                                for pretrain_learning_rate in pretrain_learning_rates:
                                    for tp_embed_dim in tp_embed_dims:
                                        for tp_layer in tp_layers:
                                            args.tp_embed_dim = tp_embed_dim
                                            args.tp_layer = tp_layer
                                            args.diff_learning_rate = diff_learning_rate
                                            args.pretrain_learning_rate = pretrain_learning_rate
                                            args.diff_embed_dim = diff_embed_dim
                                            args.diff_layer = diff_layer
                                            latent_dim_auto = latent_dim_autos[0]
                                            args.ts_learning_rate = ts_learning_rate
                                            args.embed_dim = embed_dim
                                            args.layers = layer
                                            args.cross_layers = cross_layer
                                            args.num_heads = num_head
                                            args.hidden_auto = hidden_auto
                                            args.latent_dim_auto = latent_dim_auto
                                            args.lambda0 = lambda0
                                            args.lambda1 = lambda1
                                            args.lambda2 = lambda2
                                            metrics = []
                                            for rep in range(0, args.replication):
                                                data_tr, data_val, data_te = \
                                                 load_split_data(dataset_dir,incomplete_proportion,
                                                 model_name,
                                                 bert, tokenizer,
                                                 ratio=ratio,
                                                 timestamp='hour',
                                                 temp_data=temp_data,
                                                 seed = seed**(rep+2))
                                                metrics.append(train_step(data_tr, data_val, data_te, args,
                                                                  sent_delete,tol=tol))
                                                print('metrics',metrics)
                                            metrics = np.array(metrics)
                                            Euclidean_all_mean = np.mean(metrics[:,0])
                                            Euclidean_all_std = np.std(metrics[:,0])
                                            Euclidean_complete_mean = np.mean(metrics[:,1])
                                            Euclidean_complete_std = np.std(metrics[:,1])
                                            Euclidean_incomplete_mean = np.mean(metrics[:,2])
                                            Euclidean_incomplete_std = np.std(metrics[:,2])
                                            cos_all_mean = np.mean(metrics[:,3])
                                            cos_all_std = np.std(metrics[:,3])
                                            cos_complete_mean = np.mean(metrics[:,4])
                                            cos_complete_std = np.std(metrics[:,4])
                                            cos_incomplete_mean = np.mean(metrics[:,5])
                                            cos_incomplete_std = np.std(metrics[:,5])
                                    
                                            with open(test_path, 'a') as file:
                                                params = ' lambda0= ' + str(lambda0) + ' pretrain_learning_rate = ' + str(pretrain_learning_rate)\
                                                 + ' lambda1 = ' + str(lambda1)  + ' tp_embed_dim = ' + str(tp_embed_dim)+\
                                          ' embed_dim = ' + str(embed_dim) + ' tp_layer= ' + str(tp_layer) \
                                 + ' diff_learning_rate = ' + str(diff_learning_rate) + ' diff_layer = ' + str(diff_layer)+\
                                 ' diff_embed_dim = ' + str(diff_embed_dim) + ' layer = ' + str(layer) + ' lambda2=' + str(lambda2)\
                                 + ' num_head = ' + str(num_head) + ' hidden_auto = ' + str(hidden_auto)\
                                 + ' latent_dim_auto = ' + str(latent_dim_auto)
                                                file.write(params + ' Euclidean all -- mean: ' + str(Euclidean_all_mean) + ', std: ' +\
                                       str(Euclidean_all_std) +\
                                   ' Euclidean complete -- mean: ' + str(Euclidean_complete_mean) + ', std: ' +\
                                   str(Euclidean_complete_std) + ' Euclidean incomplete -- mean: ' +\
                                   str(Euclidean_incomplete_mean) + ', std: ' + str(Euclidean_incomplete_std)+\
                                   ' Cosine_dist all -- mean: ' + str(cos_all_mean) + ', std: ' +\
                                   str(cos_all_std) + ' Cosine_dist complete -- mean: ' \
                                   + str(cos_complete_mean) + ', std: ' + str(cos_complete_std)\
                                   + ' Cosine_dist incomplete -- mean: ' + str(cos_incomplete_mean)\
                                   + ', std: ' + str(cos_incomplete_std)+'\n')
        
    

