import numpy as np
import torch
import bisect
from torch.utils.data import Dataset, DataLoader
import random
#from openai import OpenAI
import os
import requests
import json
import pickle

import warnings

warnings.filterwarnings("ignore")









    
def connect(name):
    if name.lower() == 'deepseek':
##        ## private account 

        ## organization account
        client = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "api-key": 'Your_API_Key',
    }

    if name.lower() == 'qwen':

        ## organization account
        client = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "api-key": 'Your_API_Key',
    }

    if name.lower() == 'chatgpt' or name.lower() == 'gemini':
        client = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "api-key": 'Your_API_Key',
    }

    return client




def text2token(texts, tokenizer,add_special_tokens=True,
               max_length=800):
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
##            #padding
##            token_id.extend([0] * (max_length - len(token_id)))
##            att_mask.extend([0] * (max_length - len(att_mask)))
        Textarr.append(token_id)
        Attnarr.append(att_mask)
    return Textarr, Attnarr





def load_delete_num(dataset_dir, incomplete_proportion):
    incomplete = str(int(100*incomplete_proportion))
    file = open('./'+dataset_dir+f'/incomplete{incomplete}/sent_delete.pickle'+'', 'rb')
    data = pickle.load(file)
    file.close()
    return data # {id:num sent, ...}



def get_cls(response, bert, tokenizer,max_length=512):
    # response: list of text
    results = []
    def text2token(texts, tokenizer,add_special_tokens=True,
               max_length=800):
        # texts: [strings, strings]
        max_length = 2048
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
##            #padding
##            token_id.extend([0] * (max_length - len(token_id)))
##            att_mask.extend([0] * (max_length - len(att_mask)))
            Textarr.append(token_id)
            Attnarr.append(att_mask)
        return Textarr, Attnarr
    token,attn = text2token(response, tokenizer,
                                max_length=max_length)
    #token = torch.tensor(token) # [times, 128]
    #attn = torch.tensor(attn)
    for attn_step, txts_step in zip(attn, token):
        
        txtemb = bert.bert(torch.tensor(txts_step).unsqueeze(0),
                            torch.tensor(attn_step).unsqueeze(0))\
                            [0][:,0,:].flatten()
        results.append(txtemb.tolist())
    return results
    


def text_clean(text):
    text = text.replace('**', '')
    text = text.replace('\n', ' ')
    text = text.replace('- ', '')
    for i in range(1,20):
        text = text.replace(str(i)+'. ', '')
        text = text.replace(str(i)+') ', '')
    return text


def save_pickle(data, path):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

    
def query_LLM(model_name,txt, emb_gt, incomplete_proportion,
              rep,args,save_interval=200):
    
    template = '''
The following nursing progress note may or may not have
sentences deleted. A progress note should include
sections Neuro, Cardiac, Resp, skin, GU/GI, ID, assessment, and plan. Add sentences
to the given note to make it complete if necessary. Output the note only.
------------------------------------------------------------------

'''
    
    data_all = {'emb_gt':[],'NPN':[]}
    client = connect(model_name)
    path_all = args.root + model_name + '_all' + '_incomplete' + \
               str(int(100*incomplete_proportion)) + '_rep' + str(rep) \
               + '.pickle'
    path_temp = args.root + model_name + '_temp' + '_incomplete' + \
                str(int(100*incomplete_proportion)) + '_rep' + str(rep) \
                + '.pickle'
    if os.path.isfile(path_all):
        return None

##    if os.path.isfile(path_all):
##        file = open(path_all, 'rb')
##        data_all = pickle.load(file)
##        file.close()
##        return data_all['emb_gt'], data_all['emb_inc']
    
    n = len(txt)

    while True:
        if os.path.isfile(path_temp):
            with open(path_temp, 'rb') as file:
                data_all = pickle.load(file)
            start = len(data_all['emb_gt'])
        else:
            start = 0
        idx = 0
        for emb_GT, txt_obs in zip(emb_gt[start:], txt[start:]):
            response = txt_obs
            non_empty = False if len(txt_obs)==0 else True
            # two out of 10k reports are empty
            ################# DeepSeek ########################
            if model_name == 'deepseek' and non_empty:
##                ## private account 
##                response = client.chat.completions.create(
##                    model="deepseek-reasoner",
##                    messages=[
##                    {"role": "user", "content": template + txt_obs},
##                        ],
##                    stream=False
##                    ).choices[0].message.content



                ## organization account

                
                base_url = "https://genai.hkbu.edu.hk/api/v0/rest"
                model_name1 = "deepseek-r1"
                api_version = '2024-05-01-preview'
                url = f"{base_url}/deployments/{model_name1}/chat/completions?api-version={api_version}"
                messages = [{"role": "user", "content": template + txt_obs}]
                payload = {"messages": messages, 
                           "max_tokens": 512,  "stream": False}
                
                response = requests.post(url, json=payload, headers=client).json()\
                           ['choices'][0]['message']['content'].strip()
                start=response.find('</think>\n')+9
                response = response[start:]

            ################### Qwen #######################
            elif model_name == 'qwen' and non_empty:
##                ## private account
##                response = client.chat.completions.create(
##                    model="qwen3-max",
##                    messages=[
##                    {"role": "user", "content": template + txt_obs},
##                        ],
##                    stream=False
##                    ).choices[0].message.content
                ## organization account
                base_url = "https://genai.hkbu.edu.hk/api/v0/rest"
                model_name1 = "qwen3-max"
                api_version = 'v1'
                url = f"{base_url}/deployments/{model_name1}/chat/completions?api-version={api_version}"
                messages = [{"role": "user", "content": template + txt_obs}]
                payload = {"messages": messages, 
                           "max_tokens": 512,  "stream": False}
                response = requests.post(url, json=payload, headers=client).json()\
                           ['choices'][0]['message']['content'].strip()

            ################### ChatGPT #######################
            elif model_name == 'chatgpt' and non_empty:
                base_url = "https://genai.hkbu.edu.hk/api/v0/rest"
                model_name1 = "gpt-5"
                api_version = '2024-12-01-preview'
                url = f"{base_url}/deployments/{model_name1}/chat/completions?api-version={api_version}"
                messages = [{"role": "user", "content": template + txt_obs}]
                payload = {"messages": messages, 
                           "max_tokens": 512,  "stream": False}
                response = requests.post(url, json=payload, headers=client).json()\
                           ['choices'][0]['message']['content'].strip()

            ################### Gemini #######################
            elif model_name == 'gemini' and non_empty:
                base_url = "https://genai.hkbu.edu.hk/api/v0/rest"
                model_name1 = "gemini-2.5-pro"
                api_version = 'v1'
                url = f"{base_url}/deployments/{model_name1}/chat/completions?api-version={api_version}"
                    
                messages = [{"role": "user", "content": template + txt_obs}]
                payload = {"messages": messages, 
                           "max_tokens": 512,  "stream": False}
                response = requests.post(url, json=payload, headers=client).json()\
                           ['choices'][0]['message']['content'].strip()

            
                
            response = text_clean(response)


            data_all['NPN'].append(response)
            data_all['emb_gt'].append(emb_GT)
            if idx == save_interval - 1 or idx + start == n - 1:
                break
            idx += 1

        if len(data_all['NPN']) == n:
            with open(path_all, 'wb') as file:
                pickle.dump(data_all, file)
                if os.path.isfile(path_temp):
                    os.remove(path_temp)
                break
        else:
            with open(path_temp, 'wb') as file:
                pickle.dump(data_all, file)
                print('save ', len(data_all['NPN']))



def cosine_sim(x,y):
    # x,y: [bs,seq,dim] or [bs, dim]
    dim = x.shape[-1]
    x1=x.view(-1, dim)
    y1=y.view(-1, dim)
    result = torch.sum(x1*y1,-1) / (torch.sqrt(torch.sum(x1 ** 2, -1)) \
                        * torch.sqrt(torch.sum(y1 ** 2, -1)))
    return result.flatten().tolist()



class Normalizer:
    def __init__(self, data, device='cpu'):
        # data: tensor
        self.max = data.float().max(0)[0].to(device)
        self.max.requires_grad = False
        self.min = data.float().min(0)[0].to(device)
        self.min.requires_grad = False
        self.max_min = self.max - self.min
    def __call__(self, x):
        return x
        #return ((x - self.min) / self.max_min).float()


def cal_Euclidean(x,y):
    # x,y: [bs, seq, 768] or [bs, 768]
    x1 = x.view(-1,768)
    y1 = y.view(-1,768)
    Euclidean = torch.mean(torch.sum((x1-y1)**2,-1).sqrt())
    return Euclidean.item()
    
def train_test(model_name, data_tr, data_te, bert, tokenizer, rep,args,
               sentence_dels, incomplete_proportion=0.2):

    cls_sqrt_error = 0.
    n = data_te[0]
    count = 0
    normalizer = Normalizer(torch.tensor(data_tr[1]))
    real_complete_predict, real_incomplete_predict = [],[]
    real_complete_GT, real_incomplete_GT = [],[]

    if model_name == 'naive':
        GT = normalizer(torch.tensor(data_te[1]))
        observe = normalizer(torch.tensor(data_te[2]))
        predict = observe
    else:


        path_all = args.root + model_name + '_all' + '_incomplete' + \
               str(int(100*incomplete_proportion)) + '_rep' + str(rep) \
               + '.pickle'
        file = open(path_all, 'rb')
        data_all = pickle.load(file)
        file.close()
        GT, NPN = data_all['emb_gt'], data_all['NPN']
        predict = normalizer(torch.tensor(get_cls(NPN, bert, tokenizer)))
        GT = normalizer(torch.tensor(GT))

    for idx, p_id in enumerate(data_te[3]):
        num_del = sentence_dels[p_id]
        if num_del == 0:
            real_complete_predict.append(predict[idx])
            real_complete_GT.append(GT[idx])
        else:
            real_incomplete_predict.append(predict[idx])
            real_incomplete_GT.append(GT[idx])



    

    real_complete_predict = torch.stack(real_complete_predict)
    real_complete_GT = torch.stack(real_complete_GT)
    real_incomplete_predict = torch.stack(real_incomplete_predict)
    real_incomplete_GT = torch.stack(real_incomplete_GT)
    Euclidean_all = cal_Euclidean(GT, predict)
    Cos_dist_all = (1 - np.array(cosine_sim(GT, predict))).mean()
    Euclidean_complete = cal_Euclidean(real_complete_predict, real_complete_GT)
    Cos_dist_complete = (1 - np.array(cosine_sim(real_complete_predict, real_complete_GT))).mean()
    Euclidean_incomplete = cal_Euclidean(real_incomplete_predict, real_incomplete_GT)
    Cos_dist_incomplete = (1 - np.array(cosine_sim(real_incomplete_predict, real_incomplete_GT))).mean()
    return Euclidean_all, Euclidean_complete, Euclidean_incomplete, Cos_dist_all, Cos_dist_complete,\
           Cos_dist_incomplete
                
    

def train_LLM_test(dataset_dir, incomplete_proportion,
                   model_name, test_model, bert, tokenizer,
                        args, ratio, temp_data, load_split_data, seed,
                        tol=12):

    test_path = args.root + 'synthetic' + '_' + test_model + '_incomplete' + \
               str(int(100*incomplete_proportion)) + '_evaluation.txt'
    file = open(test_path, 'w')
    file.close()

    # mortality: ts_learning_rates = [0.001,0.0001,0.00001],
    # diff_learning_rates = [0.00001,0.001] diff_embed_dims = [160],
    # diff_layers = [3]
    metrics = []
    sent_delete = load_delete_num(dataset_dir, incomplete_proportion)


    if test_model == 'naive':
        
        for rep in range(0, args.replication):
            data_tr, data_val, data_te = load_split_data(dataset_dir,incomplete_proportion,
                                                     model_name,
                                                 bert, tokenizer,
                                                 ratio=ratio,
                                                 timestamp='hour',
                                                 temp_data=temp_data,
                                                 seed = seed**(rep+2))
            metrics.append(train_test(test_model.lower(), data_tr,data_te,
                                      bert, tokenizer,rep,args,sent_delete))
    else:
        for rep in range(0, args.replication):
            data_tr, data_val, data_te = load_split_data(dataset_dir,incomplete_proportion,
                                                     model_name,
                                                 bert, tokenizer,
                                                 ratio=ratio,
                                                 timestamp='hour',
                                                 temp_data=temp_data,
                                                 seed = seed**(rep+2))
            query_LLM(test_model,data_te[0], data_te[1], incomplete_proportion, rep, args)
        for rep in range(0, args.replication):
            metrics.append(train_test(test_model.lower(), data_tr,data_te, bert,
                                      tokenizer,rep, args,sent_delete,
                                      incomplete_proportion=incomplete_proportion))
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
    print(metrics)
    with open(test_path, 'a') as file:
        file.write('Euclidean all -- mean: ' + str(Euclidean_all_mean) + ', std: ' +\
                                       str(Euclidean_all_std) +\
                   ' Euclidean complete -- mean: ' + str(Euclidean_complete_mean) + ', std: ' +\
                   str(Euclidean_complete_std) + ' Euclidean incomplete -- mean: ' +\
                   str(Euclidean_incomplete_mean) + ', std: ' + str(Euclidean_incomplete_std)+\
                   ' Cosine_dist all -- mean: ' + str(cos_all_mean) + ', std: ' +\
                   str(cos_all_std) + ' Cosine_dist complete -- mean: ' \
                   + str(cos_complete_mean) + ', std: ' + str(cos_complete_std)\
                   + ' Cosine_dist incomplete -- mean: ' + str(cos_incomplete_mean)\
                   + ', std: ' + str(cos_incomplete_std)+'\n')
    

    

    
