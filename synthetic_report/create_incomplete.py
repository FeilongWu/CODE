import os
import pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertTokenizer
import random
import torch
import numpy as np




def load_BERT(bert_name = 'ClinicalBERT',
              tokenizer_name = 'ClinicalBERT',freeze=True):
    # freeze: load BERT without backpropagation
    if not os.path.exists('./clinicalbert_cache'):
        os.mkdir('./clinicalbert_cache')

    if bert_name == 'ClinicalBERT':
        path = './clinicalbert_cache/ClinicalBERT.pickle'     
    elif bert_name == 'Longformer':        
        path = './clinicalbert_cache/LongFormer.pickle'
    elif bert_name == 'gatortron':
        path = './clinicalbert_cache/gatortron.pickle'
    bert = Base_BERT(path, bert_name,freeze=freeze)


    if tokenizer_name == 'ClinicalBERT':
        path = './clinicalbert_cache/tokenizer_clinicalbert.pickle'
    elif tokenizer_name == 'LongFormer':
        path = './clinicalbert_cache/tokenizer_longformer.pickle'
    elif bert_name == 'gatortron':
        path = './clinicalbert_cache/tokenizer_gatortron.pickle'
    tokenizer = Base_tokenizer(path, tokenizer_name)
        
    return bert, tokenizer






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



                
            
def Base_tokenizer(path, tokenizer_name):

    if os.path.isfile(path):
        with open(path, 'rb') as file:
            tokenizer = pickle.load(file)
    
    else:
        if tokenizer_name == 'ClinicalBERT':
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif tokenizer_name == 'LongFormer':
            tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        else:    
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with open(path, 'wb') as file:
            pickle.dump(tokenizer, file)
    return tokenizer





def text2token(texts, tokenizer,add_special_tokens=True,
               max_length=512):
        # texts: [strings, strings]
    max_length = 512
    Textarr = []
    Attnarr = []
    for text in texts:
        tokens = tokenizer.tokenize(text)[:max_length]
        if add_special_tokens:
            tokens = tokens[:max_length - 2]
            tokens.insert(0,'[CLS]')
            tokens.append('[SEP]')
        token_id = tokenizer.convert_tokens_to_ids(tokens)
        att_mask = [1] * len(token_id)

        token_id = np.array(token_id)
        token_id = token_id.tolist()
        #padding
        token_id.extend([0] * (max_length - len(token_id)))
        att_mask.extend([0] * (max_length - len(att_mask)))
        Textarr.append(token_id)
        Attnarr.append(att_mask)
    return Textarr, Attnarr
    


def get_CLS(data,bert,tokenizer,max_length=512):
    results = {}
    for idx in data:
        text = [data[idx]]
        token,attn = text2token(text, tokenizer,
                                   max_length=max_length)
        for attn_step, txts_step in zip(attn, token):
            txts_step = torch.tensor(txts_step)
            attn_step = torch.tensor(attn_step)
            txtemb = bert.bert(txts_step.unsqueeze(0),
                                   attn_step.unsqueeze(0))[0][:,0,:].flatten()
        results[idx] = txtemb.tolist()
    return results



    
def save_pickle(data, path):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def create_incomplete_data(data, ids, incomplete_ratio, path_complete_cls,
                           bert, tokenizer,max_length=512):
    def create_incomplete(report, num_inc=1):
        sentences_idx = []
        n = len(report)
        p = 0
        while True:
            a = p
            p = report.find('.',p,n)
            if p == -1:
                break
            p += 1
            b = p
            sentences_idx.append((a,b))
        inc_reports = []
        if len(sentences_idx) <= 1:
            for i in range(num_inc):
                if i % 2 == 0:
                    inc_reports.append(report[0:int(n/2)])
                else:
                    inc_reports.append(report[int(n/2):n])
            return inc_reports[0], 1
        sentences = []
        for i in sentences_idx:
            a, b = i
            sentences.append(report[a:b])
        n = len(sentences)
        ids = np.arange(n).tolist()
        for i in range(num_inc):
            #### delete 1-6 sentences ######
            include = max([1,n-7])
            num_include = random.randint(include,n-1) # delete 1-7 sentences
            #### delete 4 sentences ######
            #num_include = n - 4
            ## ------------------------------------------
            include_ids = sorted(random.sample(ids, num_include))
            exclude_sent = n - len(include_ids)
            r = ''
            for j in include_ids:
                a,b = sentences_idx[j]
                r += report[a:b]
            inc_reports.append(r)
        return inc_reports[0], exclude_sent
    
    with (open(path_complete_cls, "rb")) as f:
        gt_cls = pickle.load(f)
    directory = './incomplete' + str(int(100*incomplete_ratio)) + '/'
    if not os.path.isdir('./incomplete'+str(int(100*incomplete_ratio))):
        os.mkdir('incomplete'+str(int(100*incomplete_ratio)))
    inc_cls = {}
    delete_sent = {}
    n = len(ids)
    incomplete_id = ids[:int(n * incomplete_ratio)]
    reports_all = {}
    
    for idx in data:
        if idx not in incomplete_id:
            inc_cls[idx] = gt_cls[idx]
            delete_sent[idx] = 0
            reports_all[idx] = data[idx]
        else:
            inc_text, sent = create_incomplete(data[idx])
            reports_all[idx] = inc_text
            delete_sent[idx] = sent
            token,attn = text2token([inc_text], tokenizer,
                                   max_length=max_length)
            token = torch.tensor(token)
            attn = torch.tensor(attn)
            for attn_step, txts_step in zip(attn, token):
                txtemb = bert.bert(txts_step.unsqueeze(0),
                                   attn_step.unsqueeze(0))[0][:,0,:].flatten()
            inc_cls[idx] = txtemb.tolist()
    path = directory + 'CLS_inc.pickle'
    save_pickle(inc_cls, path)
    path = directory + 'sent_delete.pickle'
    save_pickle(delete_sent, path)
    path = directory + 'textual.pickle'
    save_pickle(reports_all, path)  
            

    

if __name__ == '__main__':
    complete_path = './synthetic_reports.pickle'
    with (open(complete_path, "rb")) as f:
        data = pickle.load(f)
    Clinical_BERT, tokenizer = load_BERT('ClinicalBERT', 'ClinicalBERT')
    path_complete_cls = './CLS_gt.pickle'
    if not os.path.isfile(path_complete_cls):
        complete_cls = get_CLS(data,Clinical_BERT, tokenizer)
        save_pickle(complete_cls, path_complete_cls)
    incomplete_ratio = 0.8  #### [0.2,0.4,0.6,0.8]
    ids = list(data.keys())
    random.seed(3)
    random.shuffle(ids)
    create_incomplete_data(data, ids, incomplete_ratio, path_complete_cls,
                           Clinical_BERT, tokenizer)
