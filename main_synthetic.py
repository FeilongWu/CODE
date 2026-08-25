import torch
import numpy as np
import os
import pickle
from utils_synthetic import *






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
    dataset_dir = 'synthetic_report'
    incomplete_proportion = 0.80 #[0.20,0.40,0.60,0.80]
    ratio = [0.6,0.2,0.2] # [train:validation:test]
    seed = 3
    ### Metrics ###
    # Report "Cosine_dist all"  as the metric for measuring
    # the cosine distance of generated embeddings compared
    # with ground truth
    
    model_name = 'CODE_synthetic' #[CODE_synthetic, LLM_test]
    test_model = 'naive' # ['naive','deepseek', 'qwen','chatgpt','gemini']
    if model_name == 'CODE_synthetic':
        test_model = None # ['naive','deepseek', '']
    print(('model name: ', model_name,'test_model',test_model))
    proportion_text = str(int(100*incomplete_proportion))
    temp_data = './' + model_name +  f'/incomplete{proportion_text}_tempData.pickle' # temporal processed data
    if model_name == 'CODE_synthetic':
        Clinical_BERT, tokenizer = load_BERT(bert_name='ClinicalBERT',
                                             tokenizer_name='ClinicalBERT')
        from CODE_synthetic.config import init_arg
        from CODE_synthetic.train import train_CODE_synthetic
        args = init_arg()
        train_CODE_synthetic(dataset_dir, incomplete_proportion,
                             model_name, Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
    elif model_name == 'LLM_test':
        from LLM_test.config import init_arg
        from LLM_test.train import train_LLM_test
        args = init_arg()
        #Clinical_BERT, tokenizer = None, None
        Clinical_BERT, tokenizer = load_BERT()
        train_LLM_test(dataset_dir, incomplete_proportion,
                       model_name, test_model,Clinical_BERT, tokenizer,
                            args, ratio, temp_data, load_split_data, seed)
        












