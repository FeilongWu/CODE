from MoE_Retriever.models import FusionLayer
import numpy as np
import torch
import bisect
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score





class createDS(Dataset):
    def __init__(self, data, device='cpu'):
        super(createDS, self).__init__()
        self.data = data
        self.device = device
        self.dx = len(data[0][0][0])
        self.dtxt = len(data[3][0][0])
        self.max_ts_steps = len(data[2][0])
        self.max_txt_steps = len(data[5][0])
        self.ts_pad = torch.zeros(self.dx).tolist()
        self.txt_pad = torch.zeros(self.dtxt).tolist()

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        dic = {}
        x = self.data[0][idx]
        while len(x) < self.max_ts_steps:
            x.append(self.ts_pad)
        dic['x'] = torch.tensor(x).to(self.device)
        dic['x_times'] = self.data[1][idx].unsqueeze(-1).to(self.device)
        dic['x_mask'] = self.data[2][idx].to(self.device)
        embedding = self.data[3][idx]
        while len(embedding) < self.max_txt_steps:
            embedding.append(self.txt_pad)
        dic['embedding'] = torch.tensor(embedding).to(self.device)
        dic['embedding_times'] = self.data[4][idx].unsqueeze(-1).to(self.device)
        dic['embedding_mask'] = self.data[5][idx].to(self.device)
        dic['y'] = self.data[6][idx].long().to(self.device)
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
    

@torch.no_grad()
def cal_metrics(model, data, act):
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
                    embedding_mask))[:,1]
        pred_label = torch.round(probs).flatten()
        true_y.extend(y.flatten().tolist())
        pred_y.extend(pred_label.tolist())
        pred_pro.extend(probs.flatten().tolist())
    aucroc = roc_auc_score(true_y, pred_pro)
    aucpr = average_precision_score(true_y, pred_pro)
    f1 = f1_score(true_y, pred_y)
    return f1, aucpr, aucroc
        
    

def train_step(data_tr, data_val, data_te, args, tol=12):
    args.emb_size = len(data_tr[3][0][0])
    args.time_series_size = len(data_tr[0][0][0])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # tol: tolerance for early stop
    normalizer = Normalizer(data_tr[0], args.device)
    Softmax = torch.nn.Softmax(dim=1)
    criterion = torch.nn.CrossEntropyLoss()

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
                        num_heads = args.num_heads,
                        dropout=args.dropout)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data_tr = DataLoader(createDS(data_tr, device=args.device),
                         batch_size=args.batch_size, shuffle=True)
    data_val = DataLoader(createDS(data_val, device=args.device),
                         batch_size=args.batch_size, shuffle=False)
    data_te = DataLoader(createDS(data_te, device=args.device),
                         batch_size=args.batch_size, shuffle=False)
    save_path = args.root + 'saved.pth'
    best_metrics = [0.0, 0.0, 0.0] # f1, aucpr, aucroc
    best_epoch = 0
    for epoch in range(0, args.epochs):
        model.train()
        for batch in data_tr:
            x = batch['x']
            x_times = batch['x_times']
            x_mask = batch['x_mask']
            embedding = batch['embedding']
            embedding_times = batch['embedding_times']
            embedding_mask = batch['embedding_mask']
            y = batch['y'].flatten()
            probs = model(x, x_times, x_mask, embedding, embedding_times,
                                  embedding_mask)
            task_loss = criterion(probs, y)
            #gate_loss = model.gate_loss()
            loss = task_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #### continue here
        metrics = cal_metrics(model, data_val,Softmax)
        if metric_improved(metrics, best_metrics):
            best_metrics = metrics
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch
            #print('best_epoch',best_epoch,'best_metrics',best_metrics)
        # early stop
        if (epoch - best_epoch) > tol:
            break
    # evalution
    model.load_state_dict(torch.load(save_path))
    metrics = cal_metrics(model, data_te, Softmax)
    return metrics

def train_MoE_Retriever(dataset, model_name, bert, tokenizer,
                        args, ratio, temp_data, load_split_data, seed,
                        tol=12):

    test_path = args.root + dataset + '_evaluation.txt'
    file = open(test_path, 'w')
    file.close()
    lrs = [0.00001,0.0001,0.001]
    num_layers = [2,3]
    num_experts = [16,32]
    top_ks = [2,4]
    
    for lr in lrs:
        for num_layer in num_layers:
            for num_expert in num_experts:
                for top_k in top_ks:
                    args.lr = lr
                    args.num_layers = num_layer
                    args.num_experts = num_expert
                    args.top_k = top_k
                    metrics = []
                    for rep in range(0, args.replication):
                        data_tr, data_val, data_te = load_split_data(dataset,model_name,
                                                 bert, tokenizer,
                                                 ratio=ratio,
                                                 timestamp='hour',
                                                 temp_data=temp_data,
                                                 seed = seed**(rep+1))
                        try:
                            metrics.append(train_step(data_tr, data_val, data_te, args, tol=tol))
                        except ValueError:
                            print('ValueError, continue next set')
                            continue
                    if len(metrics) < 3:
                        continue
                    else:
                        metrics = np.array(metrics)
                    f1_mean = np.mean(metrics[:,0])
                    f1_std = np.std(metrics[:,0])
                    aucpr_mean = np.mean(metrics[:,1])
                    aucpr_std = np.std(metrics[:,1])
                    aucroc_mean = np.mean(metrics[:,2])
                    aucroc_std = np.std(metrics[:,2])
                    with open(test_path, 'a') as file:
                        params = 'lr = ' + str(lr) + ' num_layer = ' + \
                             str(num_layer) + ' num_expert = ' + str(num_expert) +\
                             ' top_k = ' + str(top_k)
                        file.write(params + '    f1 score -- mean: ' + str(f1_mean) + ', std: ' +\
                               str(f1_std) +  '  AUCPR -- mean: ' + \
                               str(aucpr_mean) + ', std: ' + str(aucpr_std) + \
                               '  AUCROC -- mean: ' + str(aucroc_mean) + \
                               ', std: ' + str(aucroc_std) + '\n')
        
    
    
