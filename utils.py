import os
import pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertTokenizer
import torch
import random
import numpy as np
import bisect




class Base_BERT(torch.nn.Module):
    def __init__(self, path, bert_name,freeze=False):
        super(Base_BERT, self).__init__()
        if bert_name == 'ClinicalBERT':
            self.model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        elif bert_name == 'Longformer':
            self.model_name = 'yikuan8/Clinical-Longformer'
        elif bert_name == 'gatortron':
            # L = 24, H = 1024, A = 16
            self.model_name = 'UFNLP/gatortron-base-2k'
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                self.bert = pickle.load(file) 
        else:
            if bert_name in ['ClinicalBERT', 'gatortron']:
                self.bert = AutoModel.from_pretrained(self.model_name)
            elif bert_name == 'Longformer':
                self.bert = AutoModelForMaskedLM.from_pretrained(self.model_name)
                
            with open(path, 'wb') as file:
                pickle.dump(self.bert, file)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False



                
            
def Base_tokenizer(path, tokenizer_name):

    if os.path.isfile(path):
        with open(path, 'rb') as file:
            tokenizer = pickle.load(file)
    
    else:
        if tokenizer_name == 'ClinicalBERT':
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif tokenizer_name == 'Longformer':
            tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        elif tokenizer_name == 'gatortron':
            tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base-2k")
        with open(path, 'wb') as file:
            pickle.dump(tokenizer, file)
    return tokenizer







    
    



def load_split_data(dataset,model_name, bert,tokenizer, ratio=[6,2,2],
                    timestamp=None, seed = 3,
                    temp_data='./tempData.pickle',max_length=512,
                    endtime=48, save_interval=1000):
    # dataset: str, name of dataset
    # ratio: [tr:val:te], sum(ratio) = 1
    # timestamp: flag, convert time to hour, or normalize
    # save_interval: interval of saving intermediate data

    # path of intermediate data
    inter_data = temp_data.split('.pickle')[0] + '_intermediate.pickle'
    def split_processed(data, ratio, seed, split_fun):
        n = len(data[0])
        val_start = int(n * ratio[0])
        te_start = int(n * (ratio[0] + ratio[1]))
        id_idx = np.arange(0, n).tolist()
        random.seed(seed)
        random.shuffle(id_idx)
        return split_fun(data, val_start, te_start, id_idx)

    if os.path.isfile(temp_data):
        file = open(temp_data, 'rb')
        data_all = pickle.load(file)
        file.close()
        if model_name == 'mmtransformer':
            from mmtransformer.utils import split_data_mt
            data_tr, data_va, data_te = split_processed(data_all, ratio, seed,
                                                        split_data_mt)
        elif model_name in ['UTDE','CTPD']:
            from UTDE.utils import pre_processing, split_data_UTDE
            preprocessed = pre_processing(data_all)
            data_tr, data_va, data_te = split_processed(preprocessed, ratio, seed,
                                                        split_data_UTDE)

        elif model_name in ['FuseMoE']:
            from FuseMoE.utils import pre_processing, split_data_FuseMoE
            preprocessed = pre_processing(data_all)
            data_tr, data_va, data_te = split_processed(preprocessed, ratio, seed,
                                                        split_data_FuseMoE)
        if model_name == 'MCP':
            from MCP.utils import split_data_MCP
            data_tr, data_va, data_te = split_processed(data_all, ratio, seed,
                                                        split_data_MCP)
        if model_name == 'MoE_Retriever':
            from MoE_Retriever.utils import split_data_MoE_Retriever
            data_tr, data_va, data_te = split_processed(data_all, ratio, seed,
                                                        split_data_MoE_Retriever)

        if model_name == 'CODE':
            from CODE.utils import split_data_CODE
            data_tr, data_va, data_te = split_processed(data_all, ratio, seed,
                                                        split_data_CODE)
        elif model_name == 'AUTOFM':
            from AUTOFM.utils import split_data_AUTOFM
            data_tr, data_va, data_te = split_processed(data_all, ratio, seed,
                                                        split_data_AUTOFM)

            
            
            
        return data_tr, data_va, data_te
    path = './data/' + dataset + '.pickle'
    file = open(path, 'rb')
    data = pickle.load(file)
    ids = list(data.keys())
    n = len(ids)
    id_idx = np.arange(0, n).tolist()
    random.seed(seed)
    random.shuffle(id_idx)
    val_start = int(n * ratio[0])
    te_start = int(n * (ratio[0] + ratio[1]))
    def save_data(data, path):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()

        


    if model_name == 'mmtransformer':
        from mmtransformer.utils import load_data_mt, split_data_mt
        load_data_fun, split_data_fun = load_data_mt, split_data_mt
    elif model_name in ['UTDE','CTPD']:
        from UTDE.utils import load_data_UTDE, pre_processing, split_data_UTDE
        load_data_fun, split_data_fun = load_data_UTDE, split_data_UTDE
    elif model_name in ['FuseMoE']:
        from FuseMoE.utils import load_data_FuseMoE, pre_processing, split_data_FuseMoE
        load_data_fun, split_data_fun = load_data_FuseMoE, split_data_FuseMoE
    elif model_name == 'MCP':
        from MCP.utils import load_data_MCP, split_data_MCP
        load_data_fun, split_data_fun = load_data_MCP, split_data_MCP
    elif model_name in ['MoE_Retriever' ,'CODE']:
        max_ts_steps = 0
        max_txt_steps = 0
        max_time = endtime
        for stay_id in ids:
            max_ts_steps = max(len(data[stay_id]['dynamic'].keys()), max_ts_steps)
            max_txt_steps = max(len(data[stay_id]['notes'].keys()), max_txt_steps)

        if model_name == 'MoE_Retriever':
            from MoE_Retriever.utils import load_data_MoE_Retriever, split_data_MoE_Retriever
            load_data_fun, split_data_fun = load_data_MoE_Retriever, split_data_MoE_Retriever
        else:
            from CODE.utils import load_data_CODE, split_data_CODE
            load_data_fun, split_data_fun = load_data_CODE, split_data_CODE
        
    elif model_name == 'AUTOFM':
        from AUTOFM.utils import load_data_AUTOFM, split_data_AUTOFM
        load_data_fun, split_data_fun = load_data_AUTOFM, split_data_AUTOFM


            
        
        
    while True:
        if os.path.isfile(inter_data):
            file = open(inter_data, 'rb')
            data_all = pickle.load(file)
            file.close()
            start = len(data_all[0])
            

                
            if model_name in ['mmtransformer', 'UTDE','MCP', 'FuseMoE','AUTOFM']:
                data_all1 = load_data_fun(data, ids[start:start+save_interval],
                                          bert, tokenizer, timestamp,
                                          max_length=max_length,
                                  max_time=endtime)
            elif model_name in ['MoE_Retriever', 'CODE']:
                data_all1 = load_data_fun(data, ids[start:start+save_interval],
                                          bert, tokenizer,timestamp,
                                          max_length=max_length,
                                         max_ts_steps=max_ts_steps,
                                         max_txt_steps=max_txt_steps,
                                          max_time=max_time)

            for i in range(len(data_all)):
                if type(data_all[i]) == list:
                    data_all[i].extend(data_all1[i])
                elif torch.is_tensor(data_all[i]):
                    data_all[i] = torch.cat((data_all[i], data_all1[i]), dim=0)
        else:
                
            if model_name in ['mmtransformer', 'UTDE','MCP', 'FuseMoE','AUTOFM']:
                data_all = load_data_fun(data, ids[:save_interval],
                                          bert, tokenizer, timestamp,
                                          max_length=max_length,
                                  max_time=endtime)
            elif model_name in ['MoE_Retriever','CODE']:
                data_all = load_data_fun(data, ids[:save_interval],
                                          bert, tokenizer,timestamp,
                                          max_length=max_length,
                                         max_ts_steps=max_ts_steps,
                                         max_txt_steps=max_txt_steps,
                                         max_time=max_time)


        if len(data_all[0]) == n:
            save_data(data_all, temp_data)
            if os.path.isfile(inter_data):
                os.remove(inter_data)
            break
        else:
            save_data(data_all, inter_data)
        print(('preprocess: ', str(len(data_all[0]))))
    if model_name in ['UTDE', 'FuseMoE','CTPD']:
        data_all = pre_processing(data_all)
    data_tr, data_va, data_te = split_data_fun(data_all, val_start,
                                               te_start, id_idx)


        
        
        
        
        

    return data_tr, data_va, data_te




        
            
        
    
    
    
