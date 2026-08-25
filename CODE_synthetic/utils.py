import torch
import bisect
import numpy as np
import random






def load_data_CODE_synthetic(data, data_observed,ids, bert, tokenizer,timestamp,max_length=128,
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
               max_length=512):
        max_length = 512
        add_special_tokens = True
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

            token_id = np.array(token_id)
            token_id = token_id.tolist()
            #padding
            token_id.extend([0] * (max_length - len(token_id)))
            att_mask.extend([0] * (max_length - len(att_mask)))
            Textarr.append(token_id)
            Attnarr.append(att_mask)
        return Textarr, Attnarr
    
        


    def create_incomplete(report, num_inc=3, sent_del=1):
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
        least_sent = max([1, n-sent_del])
        for i in range(num_inc):
            num_include = random.randint(least_sent,n-1)
            include_ids = sorted(random.sample(ids, num_include))
            r = ''
            for j in include_ids:
                a,b = sentences_idx[j]
                r += report[a:b]
            inc_reports.append(r)
        return inc_reports
    
    def extract(start, end, ids, data, data_observed, timestamp,
                sent_del=7):
        print('sentences deleted: ',sent_del)
        # data: cls_gt
        # data_observed: inc text
        embedding, cls_inc, embedding_gt, p_id = [],[],[],[]
        for i in range(start, end):
            stay_id = ids[i]
            p_id.append(stay_id)
            txt_inc_temp = []
            embedding_temp = []
            embedding_gt_temp = []
            txt_temp = []
            # convert timesetps in second to <timestamp>
            ##
            
            
            embedding_gt_temp.append([data[stay_id]])
            embedding_gt.append(embedding_gt_temp)


            # Given reports
            gt_txt = data_observed[stay_id]
            if True:
                given_txt = data_observed[stay_id]
                token,attn = text2token([given_txt], tokenizer,
                                   max_length=max_length)
                token = torch.tensor(token) 
                attn = torch.tensor(attn)
                for attn_step, txts_step in zip(attn, token):
                    txtemb = bert.bert(txts_step.unsqueeze(0),
                                   attn_step.unsqueeze(0))[0][:,0,:].flatten()
                    embedding_temp.append(txtemb.tolist())
            embedding.append(embedding_temp)

                
            # Corrupted reports
            txt_inc_emb = []
            txt_inc_temp.append(create_incomplete(given_txt,sent_del=sent_del))
            for i in txt_inc_temp:
                token,attn = text2token(i, tokenizer,
                                   max_length=max_length)
                txtemb = bert.bert(torch.tensor(token), torch.tensor(attn))
                emb = txtemb[0][:,0,:]
                txt_inc_emb.append(emb.tolist())
            cls_inc.append(txt_inc_emb) # [bs, seq_len, num_inc_report, 768]
        return [embedding,cls_inc,embedding_gt,p_id]


    data_tr = extract(0, len(ids), ids, data, data_observed, timestamp)


    # data_tr; (tr_ts, tr_embd, tr_y)
    # tr_ts: [bs, timesteps, dx]
    # tr_embd: [bs, timesteps, 768]
    # tr_y: [bs,1]
    return data_tr


def pre_processing(data):
    
    # convert to tensor
    for i in range(len(data)):
        data[i] = torch.tensor(data[i])
    data[2] = data[2].squeeze(1)
    return data
        
            
def split_data_CODE_synthetic(data_all, val_start, te_start, id_idx):
    # val_start < te_start
    embedding_tr, cls_inc_tr, embedding_gt_tr,p_id_tr = [],[],[],[]
    embedding_va, cls_inc_va, embedding_gt_va,p_id_va = [],[],[],[]
    embedding_te, cls_inc_te, embedding_gt_te,p_id_te = [],[],[],[]

    for i in id_idx[0:val_start]:
        embedding_tr.append(data_all[0][i])
        cls_inc_tr.append(data_all[1][i])
        embedding_gt_tr.append(data_all[2][i])
        p_id_tr.append(data_all[3][i])


    for i in id_idx[val_start:te_start]:
        embedding_va.append(data_all[0][i])
        cls_inc_va.append(data_all[1][i])
        embedding_gt_va.append(data_all[2][i])
        p_id_va.append(data_all[3][i])

    for i in id_idx[te_start:]:
        embedding_te.append(data_all[0][i])
        cls_inc_te.append(data_all[1][i])
        embedding_gt_te.append(data_all[2][i])
        p_id_te.append(data_all[3][i])


    
    embedding_tr = torch.stack(embedding_tr)
    cls_inc_tr = torch.stack(cls_inc_tr)
    embedding_gt_tr = torch.stack(embedding_gt_tr)
    p_id_tr = torch.stack(p_id_tr)
    embedding_va = torch.stack(embedding_va)
    cls_inc_va = torch.stack(cls_inc_va)
    embedding_gt_va = torch.stack(embedding_gt_va)
    p_id_va = torch.stack(p_id_va)
    embedding_te = torch.stack(embedding_te)
    cls_inc_te = torch.stack(cls_inc_te)
    embedding_gt_te = torch.stack(embedding_gt_te)
    p_id_te = torch.stack(p_id_te)
    return (embedding_tr, cls_inc_tr, embedding_gt_tr, p_id_tr),\
           (embedding_va, cls_inc_va, embedding_gt_va, p_id_va),\
           (embedding_te, cls_inc_te, embedding_gt_te, p_id_te)

