import torch
import bisect
import numpy as np
import random






def load_data_LLM_test(cls_gt, inc_text_cls, inc_text, ids, bert,
                       tokenizer ,max_length=128,
                   max_time=48):
    # x_ts: irregular ts
    # x_ts_mask: mask of irregular ts
    # ts_tt_list = irregular ts times
    # embedding = [CLS] embedding of notes
    # note_time_list = notes times normalized to [0,1]
    # note_time_mask_list = notes times mask
    # label: binary
    # reg_ts: regular ts


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
    
        

    
    
    def extract(start, end, ids, cls_gt, inc_text_cls, inc_text):
        observe_txt, emb_gt,emb_inc,p_id = [],[],[],[]
        for i in range(start, end):
            stay_id = ids[i]
      
            observe_txt.append(inc_text[stay_id])
            emb_gt.append(cls_gt[stay_id])
            emb_inc.append(inc_text_cls[stay_id])
            p_id.append(stay_id)
                
            

        
        return [observe_txt, emb_gt,emb_inc,p_id]


    data_tr = extract(0, len(ids), ids, cls_gt, inc_text_cls, inc_text)


    # data_tr; (tr_ts, tr_embd, tr_y)
    # tr_ts: [bs, timesteps, dx]
    # tr_embd: [bs, timesteps, 768]
    # tr_y: [bs,1]
    return data_tr


def pre_processing(data):        
    return data


        
            
def split_data_LLM_test(data_all, val_start, te_start, id_idx):
    # val_start < te_start
    observe_txt_tr, emb_gt_tr,emb_inc_tr,p_id_tr = [],[],[],[]
    observe_txt_va, emb_gt_va,emb_inc_va,p_id_va = [],[],[],[]
    observe_txt_te, emb_gt_te,emb_inc_te,p_id_te = [],[],[],[]

    for i in id_idx[0:val_start]:
        observe_txt_tr.append(data_all[0][i])
        emb_gt_tr.append(data_all[1][i])
        emb_inc_tr.append(data_all[2][i])
        p_id_tr.append(data_all[3][i])

    for i in id_idx[val_start:te_start]:
        observe_txt_va.append(data_all[0][i])
        emb_gt_va.append(data_all[1][i])
        emb_inc_va.append(data_all[2][i])
        p_id_va.append(data_all[3][i])

    for i in id_idx[te_start:]:
        observe_txt_te.append(data_all[0][i])
        emb_gt_te.append(data_all[1][i])
        emb_inc_te.append(data_all[2][i])
        p_id_te.append(data_all[3][i])
    

    return (observe_txt_tr, emb_gt_tr,emb_inc_tr,p_id_tr),\
           (observe_txt_va, emb_gt_va,emb_inc_va,p_id_va),\
           (observe_txt_te, emb_gt_te,emb_inc_te,p_id_te)

