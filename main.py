import torch
import numpy as np
import os
import pickle
from utils import *




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




if __name__ == '__main__':
    
    # maximum duration
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<
    # IMPORTANT: need to manually edit "endtime" in
    # load_split_data() of './untils.py' according to follows
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<
    # mortality_data, los_data: 48hr
    # sepsis_data: 72hr
    # need to manually adjust endtime in utils.load_split_data
    # need to manually adjust time window config accordingly for
    # mmtransformer, UTDE, FuseMoE
    dataset = 'sepsis_data' # ['mortality_data','los_data',
                                      #  'sepsis_data']
    ratio = [0.6,0.2,0.2] # [train:validation:test]
    seed = 3
    Clinical_BERT, tokenizer = load_BERT(bert_name='ClinicalBERT',
                            tokenizer_name='ClinicalBERT')

    model_name = 'AUTOFM' #[mmtransformer, UTDE, FuseMoE,
                                 # MCP, MoE_Retriever, AUTOFM, SDDM,
                                 # SDDM_ts,SDDM_ts_txt,Naive,
                                 # SDDM_txt,SDDM_txt_reg,CTPD,
                                 # SDDM_original_txt, SDDM_original_multimodal]
    print(('model name: ', model_name))
    temp_data = './' + model_name + '/_' + dataset + \
                '_tempData.pickle' # temporal processed data
    if model_name == 'mmtransformer':
        from mmtransformer.config import init_arg
        from mmtransformer.train import train_mmtransformer
        args = init_arg()
        train_mmtransformer(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'UTDE':
        from UTDE.config import init_arg
        from UTDE.train import train_UTDE
        args = init_arg()
        train_UTDE(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
        
    elif model_name == 'FuseMoE':
        from FuseMoE.config import init_arg
        from FuseMoE.train import train_FuseMoE
        args = init_arg()
        train_FuseMoE(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'MCP':
        from MCP.config import init_arg
        from MCP.train import train_MCP
        args = init_arg()
        train_MCP(dataset, model_name, Clinical_BERT, tokenizer,
                  args, ratio, temp_data, load_split_data, seed)

    elif model_name == 'MoE_Retriever':
        from MoE_Retriever.config import init_arg
        from MoE_Retriever.train import train_MoE_Retriever
        args = init_arg()
        train_MoE_Retriever(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)

    elif model_name == 'AUTOFM':
        from AUTOFM.config import init_arg
        from AUTOFM.train import train_AUTOFM
        args = init_arg()
        train_AUTOFM(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'SDDM':
        from SDDM.config import init_arg
        from SDDM.train import train_SDDM
        args = init_arg()
        train_SDDM(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'SDDM_ts':
        from SDDM_ts.config import init_arg
        from SDDM_ts.train import train_SDDM_ts
        args = init_arg()
        train_SDDM_ts(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'SDDM_ts_txt':
        from SDDM_ts_txt.config import init_arg
        from SDDM_ts_txt.train import train_SDDM_ts_txt
        args = init_arg()
        train_SDDM_ts_txt(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'Naive':
        from Naive.config import init_arg
        from Naive.train import train_Naive
        args = init_arg()
        train_Naive(dataset, model_name, 
                            args, ratio, temp_data, seed)

    elif model_name == 'SDDM_txt':
        from SDDM_txt.config import init_arg
        from SDDM_txt.train import train_SDDM_txt
        args = init_arg()
        train_SDDM_txt(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'SDDM_txt_reg':
        from SDDM_txt_reg.config import init_arg
        from SDDM_txt_reg.train import train_SDDM_txt_reg
        args = init_arg()
        train_SDDM_txt_reg(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)

    elif model_name == 'CTPD':
        from CTPD.config import init_arg
        from CTPD.train import train_CTPD
        args = init_arg()
        train_CTPD(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)


    elif model_name == 'SDDM_original_txt':
        from SDDM_original_txt.config import init_arg
        from SDDM_original_txt.train import train_SDDM_original_txt
        args = init_arg()
        train_SDDM_original_txt(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)

    elif model_name == 'SDDM_original_multimodal':
        from SDDM_original_multimodal.config import init_arg
        from SDDM_original_multimodal.train import train_SDDM_original_multimodal
        args = init_arg()
        train_SDDM_original_multimodal(dataset, model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)











