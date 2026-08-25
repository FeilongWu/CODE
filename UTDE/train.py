from UTDE.models import MULTCrossModel
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

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        dic = {}
        dic['x_ts'] = self.data[0][idx].to(self.device)
        dic['x_ts_mask'] = self.data[1][idx].to(self.device)
        dic['ts_tt_list'] = self.data[2][idx].to(self.device)
        dic['embedding'] = self.data[3][idx].to(self.device)
        dic['note_time_list'] = self.data[4][idx].to(self.device)
        dic['note_time_mask_list'] = self.data[5][idx].to(self.device)
        dic['label'] = self.data[6][idx].to(self.device)
        dic['reg_ts'] = self.data[7][idx].to(self.device)
        return dic


    

    
class Normalizer:
    def __init__(self, data, device):
        # data: tensor
        self.mean = data.float().mean(0).to(device)
        self.mean.requires_grad = False
        self.std = data.float().std(0).to(device) + 0.00001
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
def cal_metrics(model, data):
    true_y, pred_y, pred_pro = [], [], []
    for batch in data:
        x_ts = batch['x_ts']
        x_ts_mask = batch['x_ts_mask']
        ts_tt_list = batch['ts_tt_list']
        embedding = batch['embedding']
        note_time_list = batch['note_time_list']
        note_time_mask_list = batch['note_time_mask_list']
        y = batch['label']
        reg_ts = batch['reg_ts']
        probs = model(x_ts, x_ts_mask, ts_tt_list, embedding, note_time_list,
                note_time_mask_list,reg_ts=reg_ts)
        pred_label = torch.round(probs).flatten()
        true_y.extend(y.flatten().tolist())
        pred_y.extend(pred_label.tolist())
        pred_pro.extend(probs.flatten().tolist())
    aucroc = roc_auc_score(true_y, pred_pro)
    aucpr = average_precision_score(true_y, pred_pro)
    f1 = f1_score(true_y, pred_y)
    return f1, aucpr, aucroc
        
    

def train_step(data_tr, data_val, data_te, args, tol=12):
    args.dx = len(data_tr[0][0][0])
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.text_seq_num = len(data_tr[3][0])
    model = MULTCrossModel(args)
    model.to(args.device)
    optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params':[p for n, p in model.named_parameters() if 'bert' in n], 'lr': \
                 args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
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
        for batch in data_tr:
            x_ts = batch['x_ts']
            x_ts_mask = batch['x_ts_mask']
            ts_tt_list = batch['ts_tt_list']
            embedding = batch['embedding']
            note_time_list = batch['note_time_list']
            note_time_mask_list = batch['note_time_mask_list']
            labels = batch['label']
            reg_ts = batch['reg_ts']
            loss = model(x_ts, x_ts_mask, ts_tt_list, embedding, note_time_list,
                note_time_mask_list,labels=labels,reg_ts=reg_ts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = cal_metrics(model, data_val)
        if metric_improved(metrics, best_metrics):
            best_metrics = metrics
            torch.save(model.state_dict(), save_path)
            best_epoch = epoch
        # early stop
        if (epoch - best_epoch) > tol:
            break
    model.load_state_dict(torch.load(save_path))
    metrics = cal_metrics(model, data_te)
    return metrics
    

def train_UTDE(dataset, model_name, bert, tokenizer,
                        args, ratio, temp_data, load_split_data, seed,
                        tol=12):

    test_path = args.root + dataset + '_evaluation.txt'
    file = open(test_path, 'w')
    file.close()
    ts_learning_rates = [0.001,0.0001,0.00001]
    txt_learning_rates = [0.001]
    embed_dims = [128, 160]
    layers = [2,3]
    cross_layers = [2,3]
    num_heads = [8]
    
    for ts_learning_rate in ts_learning_rates:
        for txt_learning_rate in txt_learning_rates:
            txt_learning_rate = ts_learning_rate
            for embed_dim in embed_dims:
                for layer in layers:
                    for cross_layer in cross_layers:
                        for num_head in num_heads:
                            args.ts_learning_rate = ts_learning_rate
                            args.txt_learning_rate = txt_learning_rate
                            args.embed_dim = embed_dim
                            args.layers = layer
                            args.cross_layers = cross_layer
                            args.num_heads = num_head
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
                                params = 'ts_learning_rate = ' + str(ts_learning_rate) + ' txt_learning_rate = ' + \
                                 str(txt_learning_rate) + ' embed_dim = ' + str(embed_dim) \
                                 + ' layer = ' + str(layer) + ' cross_layer' + str(cross_layer)\
                                 + ' num_head = ' + str(num_head)
                                file.write(params + '    f1 score -- mean: ' + str(f1_mean) + ', std: ' +\
                               str(f1_std) +  '  AUCPR -- mean: ' + \
                               str(aucpr_mean) + ', std: ' + str(aucpr_std) + \
                               '  AUCROC -- mean: ' + str(aucroc_mean) + \
                               ', std: ' + str(aucroc_std) + '\n')
        
    
    
