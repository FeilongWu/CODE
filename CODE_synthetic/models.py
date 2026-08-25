import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from CODE_synthetic.module import *
from CODE_synthetic.diffusion import *
from torch.utils.checkpoint import checkpoint










    

class MULTCrossModel(nn.Module):
    def __init__(self,args):
        super(MULTCrossModel, self).__init__()
        self.modeltype=args.modeltype
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.device = args.device
        self.kernel_size = args.kernel_size
        self.dropout=args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.tt_max=args.tt_max
        self.cross_method=args.cross_method
        self.ts_cond = args.ts_cond
        self.n_times = args.n_times
        self.latent_dim_auto = args.latent_dim_auto
        self.hidden_auto = args.hidden_auto

        self.diffusion = Diffusion(args.diff_layer, int(args.diff_embed_dim/2),
                 args.embed_dim,args.diff_embed_dim,args.dx,args.latent_dim_auto,args.hidden_auto,
                                   n_times=args.n_times,device=args.device,
                                   ts_cond=args.ts_cond,
                                   d_cls=args.d_cls, sentence_del=args.sentence_del,
                                   lambda1=args.lambda1,lambda2=args.lambda2,
                                   lambda0=args.lambda0,
                                   tp_embed_dim=args.tp_embed_dim,
                                   tp_layer=args.tp_layer)
        
        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text :
            self.time_query=torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time-1)
            self.linear = nn.Linear(1, 1)
        self.orig_d_ts=args.dx
        self.d_ts=args.embed_dim
        self.ts_seq_num=args.tt_max
        self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)

        if "TS" in self.modeltype:
            self.orig_d_ts=args.dx
            self.d_ts=args.embed_dim
            self.ts_seq_num=args.tt_max

            if self.irregular_learn_emb_ts:
                self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
 
            if self.reg_ts:
                self.orig_reg_d_ts=args.dx
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq_feature':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.orig_d_txt = args.embed_dim
            self.d_txt= args.embed_dim
            self.text_seq_num = args.text_seq_num
            #self.bertrep=BertForRepresentation(args,Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn=multiTimeAttention(args.emb_size, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = args.num_labels
        if True:
            self.trans_self_cross_ts_txt = self.get_cross_network(layers=args.cross_layers)
            dim = 0
            if "TS" in self.modeltype:
                dim += self.d_ts
            if "Text" in self.modeltype:
                dim += self.d_txt
          

            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)
            if "TS" == self.modeltype:
                self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
            elif "Text" == self.modeltype:
                self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

        if self.task=='cls':
            self.loss_fct1=nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown task")
    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len,kv_seq_len= self.d_ts,self.tt_max, None
            else:
                embed_dim, q_seq_len,kv_seq_len= self.d_ts,  self.ts_seq_num,None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim,q_seq_len,kv_seq_len= self.d_txt,self.tt_max, None
            else:
                embed_dim,q_seq_len,kv_seq_len= self.d_txt, self.text_seq_num, None

        elif self_type =='txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim,  q_seq_len,kv_seq_len= self.d_ts,self.tt_max, self.tt_max
            else:

                embed_dim,q_seq_len,kv_seq_len= self.d_ts, self.text_seq_num, self.ts_seq_num
        elif self_type =='ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len,kv_seq_len= self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len= self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                 kv_seq_len=kv_seq_len)
    def get_cross_network(self, layers=-1):
        q_seq_len = self.tt_max
        if self.modeltype == 'TS':
            embed_dim = self.d_ts
        elif self.modeltype == 'Text':
            embed_dim = self.d_txt
        return TransformerCrossEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                        q_seq_len_1=q_seq_len)



    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)



    def UpperBound(self, embedding1,
                note_time_mask_list,x_cls_inc):
        store = False
        embedding,cond, tp_densities  = self.regenerate_emb(embedding1,
                                        note_time_mask_list,store=store)
        ub = self.diffusion.upper_bound(embedding1, x_cls_inc, note_time_mask_list,
                                        embedding, tp_densities)
        return ub
        


    def forward(self, embedding1,
                note_time_mask_list,labels=None,x_cls_inc=None,
                use_checkpoint=False,pretrain=False,train=False):
        """
        dimension [batch_size, seq_len, n_features]

        """
        store = train
        if False:
            inputs = [embedding1, note_time_list,note_time_mask_list,x_ts,
                      ts_tt_list,x_ts_mask,store]
            embedding,cond, tp_densities = checkpoint(self.regenerate_emb1,inputs)
        else:
            embedding,cond, tp_densities  = self.regenerate_emb(embedding1,
                                        note_time_mask_list,store=store,
                                        pretrain=pretrain)
        if train:
            if use_checkpoint:
                inputs = [embedding1, x_cls_inc,note_time_list,
                          note_time_mask_list,x_ts, ts_tt_list,
                          cond,embedding,tp_densities]
                L_diff = checkpoint(self.diffusion.loss1, inputs)
            else:
                L_diff = self.diffusion.loss(embedding1, x_cls_inc,
                                              note_time_mask_list,
                                             embedding,tp_densities,
                                             pretrain=pretrain)
            return L_diff
        else:
            return embedding
    
        
    def regenerate_emb(self, embedding,
                            note_time_mask_list,store=False, cond=None,
                       pretrain=False):

        z, tp_densities = self.diffusion.get_encode_tp(embedding)
        if pretrain:
            x0 = None
        else:
            tp_densities = tp_densities.detach()
            x0 = self.diffusion.infer_z0(z, tp_densities, cond,\
                                                        store=store)
            
        return x0, None, tp_densities


    def regenerate_emb1(self, inputs):
        embedding, note_time_list,\
        note_time_mask_list,x_ts,ts_tt_list,\
        x_ts_mask,store = inputs
        
        return self.regenerate_emb(embedding, note_time_list,
                            note_time_mask_list,x_ts,ts_tt_list,
                       x_ts_mask,store=store)


