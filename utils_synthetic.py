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
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                self.bert = pickle.load(file) 
        else:
            if bert_name == 'ClinicalBERT':
                self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            elif bert_name == 'Longformer':
                self.bert = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")
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
        else:    
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with open(path, 'wb') as file:
            pickle.dump(tokenizer, file)
    return tokenizer







def load_split_data(dataset_dir,incomplete_ratio,
                    model_name, bert,tokenizer, ratio=[6,2,2],
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
        elif model_name in ['CODE_synthetic']:
            from CODE_synthetic.utils import pre_processing, split_data_CODE_synthetic
            preprocessed = pre_processing(data_all)
            data_tr, data_va, data_te = split_processed(preprocessed, ratio, seed,
                                                        split_data_CODE_synthetic)

        elif model_name in ['LLM_test']:
            from LLM_test.utils import pre_processing, split_data_LLM_test
            preprocessed = pre_processing(data_all)
            data_tr, data_va, data_te = split_processed(preprocessed, ratio, seed,
                                                        split_data_LLM_test)

            
            
            
        return data_tr, data_va, data_te
    # complete textual
    path = './' + dataset_dir + '/synthetic_reports.pickle'
    file = open(path, 'rb')
    complete_text = pickle.load(file)
    ids = list(complete_text.keys())
    n = len(ids)
    id_idx = np.arange(0, n).tolist()
    random.seed(seed)
    random.shuffle(id_idx)
    val_start = int(n * ratio[0])
    te_start = int(n * (ratio[0] + ratio[1]))

    # complete CLS
    path = './' + dataset_dir + '/CLS_gt.pickle'
    file = open(path, 'rb')
    cls_gt = pickle.load(file)

    # incomplete textual
    path = './' + dataset_dir + '/incomplete' +  \
           str(int(100*incomplete_ratio)) + '/textual.pickle'
    file = open(path, 'rb')
    inc_text = pickle.load(file)

    # incomplete text CLS
    path = './' + dataset_dir + '/incomplete' +  \
           str(int(100*incomplete_ratio)) + '/CLS_inc.pickle'
    file = open(path, 'rb')
    inc_text_cls = pickle.load(file)
    

    

    def save_data(data, path):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()

        


    if model_name == 'mmtransformer':
        from mmtransformer.utils import load_data_mt, split_data_mt
        load_data_fun, split_data_fun = load_data_mt, split_data_mt
    elif model_name in ['CODE_synthetic']:
        from CODE_synthetic.utils import load_data_CODE_synthetic, pre_processing, split_data_CODE_synthetic
        load_data_fun, split_data_fun = load_data_CODE_synthetic, split_data_CODE_synthetic
    elif model_name in ['LLM_test']:
        from LLM_test.utils import load_data_LLM_test, pre_processing, split_data_LLM_test
        load_data_fun, split_data_fun = load_data_LLM_test, split_data_LLM_test
       
        
        
    while True:
        if os.path.isfile(inter_data):
            file = open(inter_data, 'rb')
            data_all = pickle.load(file)
            file.close()
            start = len(data_all[0])
            

                
            if model_name in ['CODE_synthetic']:
                data_all1 = load_data_fun(cls_gt, inc_text, ids[start:start+save_interval],
                                          bert, tokenizer, timestamp,
                                          max_length=max_length,
                                  max_time=endtime)
            elif model_name in ['LLM_test']:
                data_all1 = load_data_fun(cls_gt, inc_text_cls, inc_text,
                                          ids[start:start+save_interval],
                                          bert, tokenizer,
                                          max_length=max_length)


            for i in range(len(data_all)):
                if type(data_all[i]) == list:
                    data_all[i].extend(data_all1[i])
                elif torch.is_tensor(data_all[i]):
                    data_all[i] = torch.cat((data_all[i], data_all1[i]), dim=0)
        else:
                
            if model_name in ['CODE_synthetic']:
                data_all = load_data_fun(cls_gt, inc_text,ids[:save_interval],
                                          bert, tokenizer, timestamp,
                                          max_length=max_length,
                                  max_time=endtime)
            elif model_name in ['LLM_test']:
                data_all = load_data_fun(cls_gt, inc_text_cls, inc_text,
                                          ids[:save_interval],
                                          bert, tokenizer,
                                          max_length=max_length)


        if len(data_all[0]) == n:
            save_data(data_all, temp_data)
            if os.path.isfile(inter_data):
                os.remove(inter_data)
            break
        else:
            save_data(data_all, inter_data)
        print(('preprocess: ', str(len(data_all[0]))))
    if model_name in ['CODE_synthetic', 'LLM_test']:
        data_all = pre_processing(data_all)
    data_tr, data_va, data_te = split_data_fun(data_all, val_start,
                                               te_start, id_idx)


        
        
        

    return data_tr, data_va, data_te




        
            
        
    
    
    
