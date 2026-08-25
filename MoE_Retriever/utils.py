import torch
import bisect
import numpy as np






def load_data_MoE_Retriever(data, ids, bert, tokenizer,timestamp,max_length=800,
                            max_ts_steps=10, max_txt_steps=10,max_time=50):



    def text2token(texts, tokenizer,add_special_tokens=True,
               max_length=800):
        # texts: [strings, strings]
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
            # id >= 28996 unreconized by BERT, replace by '100' unreconized text
            token_id = np.array(token_id)
            #token_id[token_id > 28995] = 100
            token_id = token_id.tolist()
            #padding
            token_id.extend([0] * (max_length - len(token_id)))
            att_mask.extend([0] * (max_length - len(att_mask)))
            Textarr.append(token_id)
            Attnarr.append(att_mask)
        return Textarr, Attnarr

    
    def convert_time(times, timestamp, min_time=0):
        # times: list of times in sec
        result = np.array(times)
        result = result - min_time
        if timestamp is None:
            return result.tolist()
        elif timestamp.lower() == 'hour':
            result = result / 3600
            return result.tolist()


    def extract(start, end, ids, data, timestamp,
                max_ts_steps=10, max_txt_steps=10,max_time=50):
        tr_ts, tr_ts_times, tr_ts_mask, tr_embd, tr_embd_times, tr_embd_mask, \
               tr_y = [],[],[],[],[],[],[]
        

        for i in range(start, end):
            stay_id = ids[i]
            tr_ts_temp = []
            tr_ts_times_temp = []
            tr_ts_mask_temp = []
            tr_txt_temp = []
            tr_txt_times_temp = []
            tr_txt_mask_temp = []
            ts_times = sorted(list(data[stay_id]['dynamic'].keys()))
            txt_times = sorted(list(data[stay_id]['notes'].keys()))
            min_time = min([ts_times[0], txt_times[0]])
            len_ts_times = len(ts_times)
            len_txt_times = len(txt_times)
            for j in ts_times:
                tr_ts_temp.append(data[stay_id]['dynamic'][j])
                tr_ts_times_temp.append(j)
            tr_ts_times_temp = convert_time(tr_ts_times_temp, timestamp, min_time=min_time)
            tr_ts_mask_temp = np.ones(len_ts_times).tolist()
            # pad ts
            for k in range(max_ts_steps - len(tr_ts_temp)):
                tr_ts_times_temp.append(tr_ts_times_temp[-1])
                tr_ts_mask_temp.append(0)

            for j in txt_times:
                tr_txt_temp.append(data[stay_id]['notes'][j])
                tr_txt_times_temp.append(j)
            tr_txt_times_temp = convert_time(tr_txt_times_temp, timestamp, min_time=min_time)
            tr_txt_mask_temp = np.ones(len_txt_times).tolist()
            token,attn = text2token(tr_txt_temp, tokenizer,
                                   max_length=max_length)
            token = torch.tensor(token)
            attn = torch.tensor(attn)
            txtemb = bert.bert(token, attn)
            tr_txt_temp = txtemb[0][:,0,:].tolist() # (NumOfNotes, 768)
            
            # pad txt
            for k in range(max_txt_steps - len(tr_txt_temp)):
                tr_txt_times_temp.append(tr_txt_times_temp[-1])
                tr_txt_mask_temp.append(0)

            tr_ts.append(tr_ts_temp)
            tr_ts_times.append(tr_ts_times_temp)
            tr_ts_mask.append(tr_ts_mask_temp)
            tr_embd.append(tr_txt_temp)
            tr_embd_times.append(tr_txt_times_temp)
            tr_embd_mask.append(tr_txt_mask_temp)
            tr_y.append(data[stay_id]['label'])

        # normalize time to max 1
        #max_time = max(max(max(tr_ts_times)), max(max(tr_embd_times)))
        tr_ts_times = (np.array(tr_ts_times) / max_time).tolist()
        tr_embd_times = (np.array(tr_embd_times) / max_time).tolist()
        return [tr_ts, torch.tensor(tr_ts_times), \
               torch.tensor(tr_ts_mask), tr_embd,\
               torch.tensor(tr_embd_times), torch.tensor(tr_embd_mask),\
               torch.tensor(tr_y)]
        
            
            
            
        
    
    data_tr = extract(0, len(ids), ids, data, timestamp,
                      max_ts_steps=max_ts_steps, max_txt_steps=max_txt_steps)


    # data_tr; (tr_ts, tr_embd, tr_y)
    # tr_ts: [bs, timesteps, dx]
    # data_tr_ts_times: [bs, timesteps]
    # data_tr_ts_mask: [bs, timesteps]
    # tr_embd: [bs, timesteps, 768]
    # tr_y: [bs,1]
    return data_tr
            
def split_data_MoE_Retriever(data_all, val_start, te_start, id_idx):
    # val_start < te_start
    data_tr_ts,data_tr_ts_times, data_tr_ts_mask, data_tr_emb, data_tr_emb_times, data_tr_emb_mask, data_tr_y = [],[],[],[],[],[],[]
    data_va_ts,data_va_ts_times, data_va_ts_mask, data_va_emb, data_va_emb_times, data_va_emb_mask, data_va_y = [],[],[],[],[],[],[]
    data_te_ts,data_te_ts_times, data_te_ts_mask, data_te_emb, data_te_emb_times, data_te_emb_mask, data_te_y = [],[],[],[],[],[],[]
    for i in id_idx[0:val_start]:
        data_tr_ts.append(data_all[0][i])
        data_tr_ts_times.append(data_all[1][i])
        data_tr_ts_mask.append(data_all[2][i])
        data_tr_emb.append(data_all[3][i])
        data_tr_emb_times.append(data_all[4][i])
        data_tr_emb_mask.append(data_all[5][i])
        data_tr_y.append(data_all[6][i])
        
    for i in id_idx[val_start:te_start]:
        data_va_ts.append(data_all[0][i])
        data_va_ts_times.append(data_all[1][i])
        data_va_ts_mask.append(data_all[2][i])
        data_va_emb.append(data_all[3][i])
        data_va_emb_times.append(data_all[4][i])
        data_va_emb_mask.append(data_all[5][i])
        data_va_y.append(data_all[6][i])

    for i in id_idx[te_start:]:
        data_te_ts.append(data_all[0][i])
        data_te_ts_times.append(data_all[1][i])
        data_te_ts_mask.append(data_all[2][i])
        data_te_emb.append(data_all[3][i])
        data_te_emb_times.append(data_all[4][i])
        data_te_emb_mask.append(data_all[5][i])
        data_te_y.append(data_all[6][i])
    return (data_tr_ts, torch.stack(data_tr_ts_times), torch.stack(data_tr_ts_mask), data_tr_emb, torch.stack(data_tr_emb_times), torch.stack(data_tr_emb_mask), torch.stack(data_tr_y)),\
           (data_va_ts, torch.stack(data_va_ts_times), torch.stack(data_va_ts_mask), data_va_emb, torch.stack(data_va_emb_times), torch.stack(data_va_emb_mask), torch.stack(data_va_y)),\
           (data_te_ts, torch.stack(data_te_ts_times), torch.stack(data_te_ts_mask), data_te_emb, torch.stack(data_te_emb_times), torch.stack(data_te_emb_mask), torch.stack(data_te_y))

