import torch
import bisect
import numpy as np
import random





def load_data_CODE(data, ids, bert, tokenizer,timestamp,max_length=512,
                            max_ts_steps=10, max_txt_steps=10,max_time=48,
                   num_inc=3):



    def text2token(texts, tokenizer,add_special_tokens=True,
               max_length=800):
        # texts: [strings, strings]
        max_length = 2048
        add_special_tokens = True
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



    def create_incomplete(report, num_inc=3, max_del=12):
        sentences_idx = []
        n = len(report)
        p = 0
        while True:
            a = p
            p = report.find('. ',p,n)
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
            return inc_reports
        sentences = []
        for i in sentences_idx:
            a, b = i
            sentences.append(report[a:b])
        n = len(sentences)
        ids = np.arange(n).tolist()
        least_sent = max([1, n-max_del])
        for i in range(num_inc):
            # delete max sentence = max_del
            num_include = random.randint(least_sent,n-1)
            include_ids = sorted(random.sample(ids, num_include))
            r = ''
            for j in include_ids:
                a,b = sentences_idx[j]
                r += report[a:b]
            inc_reports.append(r)
        return inc_reports

    

    
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
                max_ts_steps=10, max_txt_steps=10,max_time=50,
                num_inc=3):
        tr_ts, tr_ts_times, tr_ts_mask, tr_embd, tr_embd_times, tr_embd_mask, \
               cls_inc, tr_y,p_id = [],[],[],[],[],[],[],[],[]
        for time in list(data[ids[0]]['dynamic'].keys()):
            dx = len(data[ids[0]]['dynamic'][time])
            break
        ts_pad = [0.] * dx
        emb_pad = [0.] * 1024
        emb_inc_pad = [emb_pad for i in range(num_inc)]
        
        
            
        

        for i in range(start, end):
            stay_id = ids[i]
            p_id.append(stay_id)
            tr_ts_temp = []
            tr_ts_times_temp = []
            tr_ts_mask_temp = []
            tr_txt_temp = []
            tr_txt_times_temp = []
            tr_txt_mask_temp = []
            txt_inc_temp = []
            ts_times = sorted(list(data[stay_id]['dynamic'].keys()))
            txt_times = sorted(list(data[stay_id]['notes'].keys()))
            len_ts_times = len(ts_times)
            len_txt_times = len(txt_times)
            for j in ts_times:
                tr_ts_temp.append(data[stay_id]['dynamic'][j])
                tr_ts_times_temp.append(j)
            min_time = min([ts_times[0], txt_times[0]])
            tr_ts_times_temp = convert_time(tr_ts_times_temp, timestamp,
                                            min_time=min_time)
            tr_ts_mask_temp = np.ones(len_ts_times).tolist()
            # pad ts
            tr_ts_temp_len = len(tr_ts_temp)
            for k in range(max_ts_steps - tr_ts_temp_len):
                tr_ts_times_temp.append(tr_ts_times_temp[-1])
                tr_ts_mask_temp.append(0)
                tr_ts_temp.append(ts_pad)

            for j in txt_times:
                txt = data[stay_id]['notes'][j]
                tr_txt_temp.append(txt)
                tr_txt_times_temp.append(j)
                txt_inc_temp.append(create_incomplete(txt,num_inc=num_inc))
            tr_txt_times_temp = convert_time(tr_txt_times_temp, timestamp,
                                             min_time=min_time)
            tr_txt_mask_temp = np.ones(len_txt_times).tolist()
            token,attn = text2token(tr_txt_temp, tokenizer,
                                   max_length=max_length)
            tr_txt_temp = []
            for attn_step, txts_step in zip(attn, token):
                txts_step = torch.tensor(txts_step)
                attn_step = torch.tensor(attn_step)
                txtemb = bert.bert(txts_step.unsqueeze(0),
                                   attn_step.unsqueeze(0))[0][:,0,:].flatten()
                tr_txt_temp.append(txtemb.tolist()) # (NumOfNotes, 1024)

            # incomplete reports
            txt_inc_emb = []
            
            for i in txt_inc_temp:
                token,attn = text2token(i, tokenizer,
                                   max_length=max_length)
                txt_inc_emb_temp = []
                for token_step, attn_step in zip(token,attn):
                    txtemb = bert.bert(torch.tensor([token_step]),
                                       torch.tensor([attn_step]))[0][:,0,:]
                    txt_inc_emb_temp.append(txtemb.flatten().tolist())
                txt_inc_emb.append(txt_inc_emb_temp)
            


            
            
            
            # pad txt
            tr_txt_temp_len = len(tr_txt_temp)
            for k in range(max_txt_steps - tr_txt_temp_len):
                tr_txt_times_temp.append(tr_txt_times_temp[-1])
                tr_txt_mask_temp.append(0)
                tr_txt_temp.append(emb_pad)
                txt_inc_emb.append(emb_inc_pad)
                
                

            tr_ts.append(tr_ts_temp)
            tr_ts_times.append(tr_ts_times_temp)
            tr_ts_mask.append(tr_ts_mask_temp)
            tr_embd.append(tr_txt_temp)
            tr_embd_times.append(tr_txt_times_temp)
            tr_embd_mask.append(tr_txt_mask_temp)
            tr_y.append(data[stay_id]['label'])
            cls_inc.append(txt_inc_emb)

        # normalize time to max 1
        #max_time = max(max(max(tr_ts_times)), max(max(tr_embd_times)))
        tr_ts_times = (np.array(tr_ts_times) / max_time).tolist()
        tr_embd_times = (np.array(tr_embd_times) / max_time).tolist()
        return [torch.tensor(tr_ts), torch.tensor(tr_ts_times), \
               torch.tensor(tr_ts_mask), torch.tensor(tr_embd),\
               torch.tensor(tr_embd_times), torch.tensor(tr_embd_mask),\
               torch.tensor(tr_y), torch.tensor(cls_inc), torch.tensor(p_id)]
        
            
            
            
        
    
    data_tr = extract(0, len(ids), ids, data, timestamp,
                      max_ts_steps=max_ts_steps, max_txt_steps=max_txt_steps,
                      num_inc=num_inc, max_time=max_time)


    # data_tr; (tr_ts, tr_embd, tr_y)
    # tr_ts: [bs, timesteps, dx]
    # data_tr_ts_times: [bs, timesteps]
    # data_tr_ts_mask: [bs, timesteps]
    # tr_embd: [bs, timesteps, 1024]
    # tr_y: [bs,1]
    return data_tr
            
def split_data_CODE(data_all, val_start, te_start, id_idx):
    # val_start < te_start
    data_tr_ts,data_tr_ts_times, data_tr_ts_mask, data_tr_emb, \
    data_tr_emb_times, data_tr_emb_mask, data_tr_y,cls_inc_tr = [],[],[],[],[],[],[],[]
    data_va_ts,data_va_ts_times, data_va_ts_mask, data_va_emb, \
    data_va_emb_times, data_va_emb_mask, data_va_y,cls_inc_va = [],[],[],[],[],[],[],[]
    data_te_ts,data_te_ts_times, data_te_ts_mask, data_te_emb, \
    data_te_emb_times, data_te_emb_mask, data_te_y,cls_inc_te = [],[],[],[],[],[],[],[]
    for i in id_idx[0:val_start]:
        data_tr_ts.append(data_all[0][i])
        data_tr_ts_times.append(data_all[1][i])
        data_tr_ts_mask.append(data_all[2][i])
        data_tr_emb.append(data_all[3][i])
        data_tr_emb_times.append(data_all[4][i])
        data_tr_emb_mask.append(data_all[5][i])
        data_tr_y.append(data_all[6][i])
        cls_inc_tr.append(data_all[7][i])

        
    for i in id_idx[val_start:te_start]:
        data_va_ts.append(data_all[0][i])
        data_va_ts_times.append(data_all[1][i])
        data_va_ts_mask.append(data_all[2][i])
        data_va_emb.append(data_all[3][i])
        data_va_emb_times.append(data_all[4][i])
        data_va_emb_mask.append(data_all[5][i])
        data_va_y.append(data_all[6][i])
        cls_inc_va.append(data_all[7][i])


    for i in id_idx[te_start:]:
        data_te_ts.append(data_all[0][i])
        data_te_ts_times.append(data_all[1][i])
        data_te_ts_mask.append(data_all[2][i])
        data_te_emb.append(data_all[3][i])
        data_te_emb_times.append(data_all[4][i])
        data_te_emb_mask.append(data_all[5][i])
        data_te_y.append(data_all[6][i])
        cls_inc_te.append(data_all[7][i])

    return (torch.stack(data_tr_ts), torch.stack(data_tr_ts_times), torch.stack(data_tr_ts_mask), \
            torch.stack(data_tr_emb), torch.stack(data_tr_emb_times), torch.stack(data_tr_emb_mask), \
            torch.stack(data_tr_y), torch.stack(cls_inc_tr)),\
           (torch.stack(data_va_ts), torch.stack(data_va_ts_times), torch.stack(data_va_ts_mask), \
            torch.stack(data_va_emb), torch.stack(data_va_emb_times), torch.stack(data_va_emb_mask), \
            torch.stack(data_va_y), torch.stack(cls_inc_va)),\
           (torch.stack(data_te_ts), torch.stack(data_te_ts_times), torch.stack(data_te_ts_mask), \
            torch.stack(data_te_emb), torch.stack(data_te_emb_times), torch.stack(data_te_emb_mask), \
            torch.stack(data_te_y), torch.stack(cls_inc_te))
