import torch
import bisect
import numpy as np






def load_data_UTDE(data, ids, bert, tokenizer,timestamp,max_length=128,
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
    
        


    class Normalizer:
        def __init__(self,data):
            data1 = np.array(data)
            self.mean = data1.mean(0)
            self.std = data1.std(0) + 0.00001
        def __call__(self,x):
            return (x - self.mean) / self.std

    def convert_time(times, timestamp, min_time=0):
        # times: list of times in sec
        result = np.array(times)
        result = result - min_time
        if timestamp is None:
            return result.tolist()
        elif timestamp.lower() == 'hour':
            result = result / 3600
            return result.tolist()
    
    def extract(start, end, ids, data, timestamp, normalizer):
        x_ts, x_ts_mask, ts_tt_list, embedding, note_time_list,\
        note_time_mask_list, label, reg_ts = [],[],[],[],[],[],[],[]
        for i in range(start, end):
            stay_id = ids[i]
            reg_ts_temp = []
            irg_ts_temp = []
            ts_mask_temp = []
            ts_time_temp = []
            embedding_temp = []
            note_time_temp = []
            note_time_mask_temp = []
            txt_temp = []
            ts_times = sorted(list(data[stay_id]['dynamic'].keys()))
            txt_times = sorted(list(data[stay_id]['notes'].keys()))
            min_time = min([ts_times[0], txt_times[0]])
            # convert timesetps in second to <timestamp>
            ts_time_temp1 = convert_time(ts_times, timestamp,min_time=min_time)
            note_time_temp1 = convert_time(txt_times, timestamp,min_time=min_time)
            ##
            len_ts_times = len(ts_times)
            for j in np.arange(0,max_time):
                idx = bisect.bisect_left(ts_times, j)
                if idx < len_ts_times:
                    reg_ts_temp.append(normalizer(np.array(data[stay_id]['dynamic'][ts_times[idx]])).tolist())
                else:
                    reg_ts_temp.append(normalizer(np.array(data[stay_id]['dynamic'][ts_times[-1]])).tolist())
            reg_ts.append(reg_ts_temp)


            for idx,j in enumerate(ts_times):
                if ts_time_temp1[idx] > max_time:
                    break
                temp = data[stay_id]['dynamic'][j]
                irg_ts_temp.append(normalizer(np.array(temp)).tolist())
                ts_mask_temp.append(np.ones(len(temp)).tolist())
                ts_time_temp.append(ts_time_temp1[idx])

            x_ts.append(irg_ts_temp)
            x_ts_mask.append(ts_mask_temp)
                
            for idx,j in enumerate(txt_times):
                if note_time_temp1[idx] > max_time:
                    break
                txt = data[stay_id]['notes'][j] # [str]
                txt_temp.append(txt)
                note_time_temp.append(note_time_temp1[idx])
                
            if len(txt_temp) == 0:
                txt_temp.append(data[stay_id]['notes'][txt_times[0]])
                note_time_temp.append([max_time])
                
            
            ts_tt_list.append((np.array(ts_time_temp) / max_time).tolist())
            note_time_list.append((np.array(note_time_temp) / max_time).tolist())
            
            token,attn = text2token(txt_temp, tokenizer,
                                   max_length=max_length)
            token = torch.tensor(token) # [times, 128]
            attn = torch.tensor(attn)
            for attn_step, txts_step in zip(attn, token):
                txtemb = bert.bert(txts_step.unsqueeze(0),
                                   attn_step.unsqueeze(0))[0][:,0,:].flatten()
                #txtemb: [768]
                embedding_temp.append(txtemb.tolist())
                note_time_mask_temp.append(np.ones(768).tolist())
            embedding.append(embedding_temp)
            note_time_mask_list.append(note_time_mask_temp)
            label.append(data[stay_id]['label'])
        return [x_ts, x_ts_mask, ts_tt_list, embedding, note_time_list,\
               note_time_mask_list, label, reg_ts]

    ts = []
    for i in data:
        for j in data[i]['dynamic']:
            ts.append(data[i]['dynamic'][j])
    normalizer = Normalizer(ts)
    del ts
    data_tr = extract(0, len(ids), ids, data, timestamp, normalizer)


    # data_tr; (tr_ts, tr_embd, tr_y)
    # tr_ts: [bs, timesteps, dx]
    # tr_embd: [bs, timesteps, 768]
    # tr_y: [bs,1]
    return data_tr


def pre_processing(data):
    ts_longest = 0
    emb_longest = 0
    for i in data[0]:
        temp = len(i) # num of times
        if temp > ts_longest:
            ts_longest = temp
    dx = len(i[0])
    for i in data[3]:
        temp = len(i)
        if temp > emb_longest:
            emb_longest = temp
    d_emb = len(data[3][0][0])
    total = len(data[0])
    # pad to same num of times
    for i in range(total):
        x_ts = data[0][i]
        n = len(x_ts)
        x_ts.extend(np.zeros((ts_longest - n, dx)).tolist())
        x_ts_mask = data[1][i]
        x_ts_mask.extend(np.zeros((ts_longest - n, dx)).tolist())
        ts_tt_list = data[2][i]
        ts_tt_list.extend(np.ones(ts_longest - n).tolist())

        embedding = data[3][i]
        n = len(embedding)
        embedding.extend(np.zeros((emb_longest - n, d_emb)).tolist())
        note_time_list = data[4][i]
        note_time_list.extend(np.ones(emb_longest - n).tolist())
        note_time_mask_list = data[5][i]
        note_time_mask_list.extend(np.zeros((emb_longest - n, d_emb)).tolist())
    # convert to tensor
    return [torch.tensor(i) for i in data]
        
            
def split_data_UTDE(data_all, val_start, te_start, id_idx):
    # val_start < te_start
    x_ts_tr, x_ts_mask_tr, ts_tt_list_tr, embedding_tr, note_time_list_tr,\
    note_time_mask_list_tr, label_tr, reg_ts_tr = [],[],[],[],[],[],[],[]
    x_ts_va, x_ts_mask_va, ts_tt_list_va, embedding_va, note_time_list_va,\
    note_time_mask_list_va, label_va, reg_ts_va = [],[],[],[],[],[],[],[]
    x_ts_te, x_ts_mask_te, ts_tt_list_te, embedding_te, note_time_list_te,\
    note_time_mask_list_te, label_te, reg_ts_te = [],[],[],[],[],[],[],[]

    for i in id_idx[0:val_start]:

        x_ts_tr.append(data_all[0][i])
        x_ts_mask_tr.append(data_all[1][i])
        ts_tt_list_tr.append(data_all[2][i])
        embedding_tr.append(data_all[3][i])
        note_time_list_tr.append(data_all[4][i])
        note_time_mask_list_tr.append(data_all[5][i])
        label_tr.append(data_all[6][i])
        reg_ts_tr.append(data_all[7][i])

    for i in id_idx[val_start:te_start]:
        x_ts_va.append(data_all[0][i])
        x_ts_mask_va.append(data_all[1][i])
        ts_tt_list_va.append(data_all[2][i])
        embedding_va.append(data_all[3][i])
        note_time_list_va.append(data_all[4][i])
        note_time_mask_list_va.append(data_all[5][i])
        label_va.append(data_all[6][i])
        reg_ts_va.append(data_all[7][i])

    for i in id_idx[te_start:]:
        x_ts_te.append(data_all[0][i])
        x_ts_mask_te.append(data_all[1][i])
        ts_tt_list_te.append(data_all[2][i])
        embedding_te.append(data_all[3][i])
        note_time_list_te.append(data_all[4][i])
        note_time_mask_list_te.append(data_all[5][i])
        label_te.append(data_all[6][i])
        reg_ts_te.append(data_all[7][i])
    

    return (x_ts_tr, x_ts_mask_tr, ts_tt_list_tr, embedding_tr, note_time_list_tr,\
    note_time_mask_list_tr, label_tr, reg_ts_tr),\
           (x_ts_va, x_ts_mask_va, ts_tt_list_va, embedding_va, note_time_list_va,\
    note_time_mask_list_va, label_va, reg_ts_va),\
           (x_ts_te, x_ts_mask_te, ts_tt_list_te, embedding_te, note_time_list_te,\
    note_time_mask_list_te, label_te, reg_ts_te)

