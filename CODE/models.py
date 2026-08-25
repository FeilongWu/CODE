
from CODE.module import *
from CODE.diffusion import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter



eps = 1e-06


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)






class FusionLayer(nn.Module):
    def __init__(self, num_modalities, emb_size, time_series_size, hidden_dim,
                 output_dim, num_layers, num_layers_pred, num_experts,
                 num_routers, top_k, normalizer, args,\
                 num_heads=2, dropout=0.5, mlp_sparse=False):
        super(FusionLayer, self).__init__()
        self.emb_size = emb_size
        self.device = args.device
        self.dx = time_series_size
        layers = []
        layers2 = []
        layers.append(TransformerEncoderLayer(num_experts, num_routers,
                                              hidden_dim, num_head=num_heads,
                                              dropout=dropout, hidden_times=2,
                                              mlp_sparse=mlp_sparse, top_k=top_k))
##        layers2.append(TransformerEncoderLayer(num_experts, num_routers,
##                                              hidden_dim, num_head=num_heads,
##                                              dropout=dropout, hidden_times=2,
##                                              mlp_sparse=mlp_sparse, top_k=top_k))
        for j in range(num_layers - 1):
            tmp = (mlp_sparse) & (j % 2 == 1)
            layers.append(TransformerEncoderLayer(num_experts, num_routers,
                                                  hidden_dim, num_head=num_heads,
                                                  dropout=dropout, hidden_times=2,
                                                  mlp_sparse=tmp, top_k=top_k))
##            layers2.append(TransformerEncoderLayer(num_experts, num_routers,
##                                                  hidden_dim, num_head=num_heads,
##                                                  dropout=dropout, hidden_times=2,
##                                                  mlp_sparse=tmp, top_k=top_k))
            
        self.last_layer = MLP(hidden_dim*num_modalities, hidden_dim,
                              output_dim, num_layers_pred,
                              activation=nn.ReLU(), dropout=0.5)
        self.num_heads = num_heads
        self.normalizer = normalizer
        
        self.network = nn.Sequential(*layers)
##        self.network2 = nn.Sequential(*layers2)
        self.pos_embed = MLP(1, hidden_dim, hidden_dim, 1, dropout=0.0)
        self.ts_encoder = MLP(time_series_size, hidden_dim, hidden_dim, num_layers)
        self.diffusion = Diffusion(args.num_layer, int(args.diff_embed_dim/2),
                 args.embed_dim,args.diff_embed_dim,args.time_series_size,n_times=args.n_times,
                                   device=args.device,
                                   ts_cond=args.ts_cond,
                                   lambda1=args.lambda1,
                                   lambda0=args.lambda0,
                                   highest_percent=args.highest_percent,
                                   tp_embed_dim=args.tp_embed_dim,
                                   tp_layer=args.tp_layer)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.ts_cond = args.ts_cond
        self.multimodal = args.multimodal
        self.use_diffusion = args.use_diffusion
        self.x0 = None
        self.tp_densities = None
        self.cond = None
        if args.ts_cond:
            self.periodic = nn.Linear(1, args.embed_time-1)
            self.linear = nn.Linear(1, 1)
            self.time_attn_ts=multiTimeAttention(args.time_series_size*2, args.embed_dim, \
                                                 args.embed_time, 8)
        


        
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)


    def pad_to_mask(self, pad_indicate):
        # pad_indicate: [bs, num times]
        # debug for self-attention, no gaurantee for cross-attn
        pad_indicate = pad_indicate.unsqueeze(-1)
        return torch.bmm(pad_indicate,pad_indicate.transpose(1, 2)).repeat(self.num_heads, 1, 1)


    def loss(self, ts, ts_times, ts_mask, emb, emb_times, emb_mask,
             y, x_cls_inc, pretrain=False):
        #pretrain:  MoE-retriever

        
        probs = self.forward(ts, ts_times, ts_mask, emb, emb_times, emb_mask,
                             pretrain=pretrain)
        loss = self.criterion(probs, y)
        if not pretrain:
            loss += self.diffusion.loss(emb, x_cls_inc, emb_mask, self.x0,
                                        self.tp_densities, self.cond,
                                        )
        return loss
        



    
    def forward(self, ts, ts_times, ts_mask, emb, emb_times, emb_mask,
                pretrain=False, testing=False):
        ts_mask1 = ts_mask
        if self.multimodal:
            ts = self.normalizer(ts)
            ts_emb = self.ts_encoder(ts)
            ts_emb += self.pos_embed(ts_times)
            ts_mask = self.pad_to_mask(ts_mask)
            for i in range(len(self.network)):
                ts_emb = self.network[i](ts_emb, attn_mask=ts_mask)
        
        if not pretrain and self.use_diffusion == True:
            bs, seq, _ = emb.shape
            emb_gen = self.regenerate_emb(emb, emb_times, emb_mask,
                                      ts, ts_times, ts_mask1,
                                      store=not testing)
            p_tp_0 = self.tp_densities.view(-1,self.diffusion.total_times)\
                     [:,0].unsqueeze(-1)
            tp_0_mask = self.diffusion.rank_mask(p_tp_0, emb_mask,\
                                    self.diffusion.highest_percent).view(bs, seq, 1)
            emb = (tp_0_mask * emb + (1 - tp_0_mask) * \
                  emb_gen).view(bs, seq, self.emb_size)
        emb += self.pos_embed(emb_times)

        
        emb_mask = self.pad_to_mask(emb_mask)
        for i in range(len(self.network)):
            #emb = self.network2[i](emb, attn_mask=emb_mask)
            emb = self.network[i](emb, attn_mask=emb_mask)
        # note if ts_emb dimension is [bs, num times, 1024], need to consider mask
        # for calculating mean
        
        if self.multimodal:
            x = self.last_layer(torch.cat([ts_emb.mean(dim=1),
                                       emb.mean(dim=1)], dim=1))
        else:
            x = self.last_layer(emb.mean(dim=1))
        
        return x

    
    def regenerate_emb(self, embedding, note_time_list,
                       note_time_mask_list, x_ts, ts_tt_list,
                       x_ts_mask,store=False):
        if self.ts_cond == True:
            x_ts_mask1 = x_ts_mask.unsqueeze(-1).repeat(1,1,self.dx)
            time_query = self.learn_time_embedding(note_time_list)
            time_key_ts = self.learn_time_embedding(ts_tt_list)
            cond = self.time_attn_ts(time_query, time_key_ts,
                                     torch.cat((x_ts,x_ts_mask1), 2),
                                     torch.cat((x_ts_mask1,x_ts_mask1), 2))
            bs,seq,dim = cond.shape
            cond = cond.view(bs*seq, dim)
        else:
            cond = None
        if store:
            bs,seq_len,_ = embedding.shape
            self.diffusion.bs = bs
            self.diffusion.seq_len = seq_len
            bs_seq_len = bs*seq_len
            t_p = torch.randint(low=0, high=self.diffusion.total_times,
                            size=[bs, seq_len]).to(self.device).view(bs_seq_len,1)
            self.diffusion.t_p = t_p
            self.diffusion.init_params_ELBO_0_tp(bs_seq_len)
        z, tp_densities = self.diffusion.get_encode_tp(embedding)
        tp_densities = tp_densities.detach()
        x0 = self.diffusion.infer_z0(z, tp_densities, cond,\
                                                        store=store)
        self.x0 = x0
        self.tp_densities = tp_densities
        self.cond = cond
        return x0






class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                num_experts,
                num_routers,
                d_model, 
                num_head, 
                dropout=0.1, 
                activation=nn.GELU, 
                hidden_times=2, 
                mlp_sparse = False, 
                self_attn = True,
                top_k=2,
                **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(
            d_model, num_head,  attn_dropout=dropout, proj_drop=dropout)
        
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn

        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, d_hidden=d_model * hidden_times, activation=nn.GELU(), top_k=top_k, **kwargs)
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=nn.GELU(), dropout=dropout)

    def forward(self, x, attn_mask=None, expert_index=None):
        # x: [bs, seq_len, dim]
        if self.self_attn:
            if expert_index:
                x = self.attn(x, x, x, attn_mask=attn_mask)
                x = x + self.dropout1(x)
                x = self.mlp(self.norm2(x), expert_index)
                return x
            
            else:
                #chunk_size = [item.shape[1] for item in x]
                x = self.norm1(x).transpose(1,0)
                x = self.attn(x, x, x, attn_mask=attn_mask)
                x = x + self.dropout1(x)
                #x = torch.split(x, chunk_size, dim=1)
                #x = [item for item in x]

                if self.mlp_sparse:
                    #for i in range(len(chunk_size)):
                    #    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), expert_index))
                    x = x + self.dropout2(self.mlp(self.norm2(x), expert_index))
                else:
                    #for i in range(len(chunk_size)):
                    #    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
                    x = x + self.dropout2(self.mlp(self.norm2(x)))
        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m, other_m)
            x = [x[i]+self.dropout1(x[i]) for i in range(len(chunk_size))]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        return x.transpose(1,0)




    

class Attention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False,
                 proj_drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.proj_drop = proj_drop

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def masked_att(self, weight, mask):
        weight1 = (torch.exp(weight.float()) * mask).type_as(weight)
        w_sum = torch.sum(weight1,dim=-1).unsqueeze(-1) + eps
        return weight1 / w_sum

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, only binary mask, 0 for padded.
        """

        # import pdb;
        # pdb.set_trace()

        if attn_mask is None:
            attn_mask = torch.ones(query.shape[0], query.shape[1], key.shape[1])\
                        .repeat(self.num_heads,1,1)
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]


##        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = self.masked_att(attn_weights, attn_mask)

        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn = F.dropout(attn, p=self.proj_drop, training=self.training)
        

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
    



