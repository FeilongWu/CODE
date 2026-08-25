'''
best true positive: 0.48
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np






class Autoencoder(nn.Module):
    def __init__(self, input_d, hidden, latent_dim, n_layers, act='ReLU'):
        super().__init__()
        # n_layers: number of total layers
        act = find_act(act)
        def Layers(input_d, layers):
            results = []
            for i,j in enumerate(layers):
                if i == 0:
                    results.append((input_d, j))
                else:
                    results.append((layers[i-1], j))
            sequence = []
            for i in results:
                sequence.append(nn.Linear(i[0],i[1]))
                sequence.append(act)
            return sequence[:-1]
                
            
        encoder_layers = [int(i) for i in np.linspace(hidden, latent_dim, n_layers)]
        self.encoder = nn.Sequential(*Layers(input_d,encoder_layers))
        reverse_layers =list(reversed(encoder_layers))
        reverse_layers.append(input_d)
        #for i in reversed(encoder_layers):
        #    reverse_layers.append((i[1],i[0]))
        
        self.decoder = nn.Sequential(*Layers(latent_dim,reverse_layers[1:]))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)



    

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb






def find_act(act):
    # act: function name
    if act.lower() == 'relu':
        actv = nn.ReLU()
    elif act.lower() == 'sigmoid':
        actv = nn.Sigmoid()
    elif act.lower() == 'softplus':
        actv = nn.Softplus()
    elif act.lower() == 'elu':
        actv = nn.ELU()
    elif act.lower() == 'silu':
        actv = nn.SiLU()
    return actv





class t_prime_predictor(nn.Module):
    def __init__(self, input_dim, out_dim, latent_dim, n_layers,
                 act='ReLU'):
        super().__init__()
        self.act = find_act(act)
        layers = []
        layers.append(nn.Linear(input_dim, latent_dim))
        layers.append(self.act)
        for i in range(n_layers):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(self.act)
        layers.append(nn.Linear(latent_dim, out_dim))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class Denoise_layer(nn.Module):
    def __init__(self, in_dim, out_dim, act='SiLU',residual=True,
                 ts_cond=True):
        super().__init__()
        if act:
            self.act = find_act(act)
        else:
            act = None
        self.layer = nn.Linear(in_dim, out_dim)
        if ts_cond:
            self.forward = self.forward1 if act else self.forward2
        else:
            self.forward = self.forward3 if act else self.forward4
        
        self.residual = residual

    def forward1(self, x, cond, t_emd):
        # with activation
        out = self.layer(torch.cat((x, cond, t_emd), dim=1))
        if self.residual:
            return self.act(out + x)
        else:
            return self.act(out)

    def forward2(self, x, cond, t_emd):
        # without activation
        out = self.layer(torch.cat((x, cond, t_emd), dim=1))
        if self.residual:
            return out + x
        else:
            return out


    def forward3(self, x, cond, t_emd):
        # cond=None
        # with activation
        out = self.layer(torch.cat((x, t_emd), dim=1))
        if self.residual:
            return self.act(out + x)
        else:
            return self.act(out)

    def forward4(self, x, cond, t_emd):
        # cond=None
        # without activation
        out = self.layer(torch.cat((x, t_emd), dim=1))
        if self.residual:
            return out + x
        else:
            return out


class Denoiser(nn.Module):
    
    def __init__(self, input_dim, cond_dim, latent_dim, n_layers,
                 diffusion_time_embedding_dim = 256, n_times=1000,
                 ts_cond=True):
        # n_layers: num of hidden layers
        super(Denoiser, self).__init__()
        if ts_cond == True:
        #self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
            all_dim = input_dim + cond_dim + diffusion_time_embedding_dim
            self.in_project = Denoise_layer(all_dim,
                                        latent_dim, act=None, residual=False)
        
            all_dim = latent_dim + cond_dim + diffusion_time_embedding_dim
            self.denoising = nn.ModuleList([Denoise_layer(all_dim,
                                                      latent_dim) for i in range(n_layers)])
            self.out_project = Denoise_layer(all_dim,
                                        input_dim, act=None, residual=False)
        if ts_cond == False:
            all_dim = input_dim + diffusion_time_embedding_dim
            self.in_project = Denoise_layer(all_dim,
                                        latent_dim, act=None, residual=False,
                                            ts_cond=ts_cond)
        
            all_dim = latent_dim + diffusion_time_embedding_dim
            self.denoising = nn.ModuleList([Denoise_layer(all_dim,latent_dim
                                                      ,ts_cond=ts_cond) for i in range(n_layers)])
            self.out_project = Denoise_layer(all_dim,
                                        input_dim, act=None, residual=False,
                                             ts_cond=ts_cond)
            
        

        
        
    def forward(self, x, cond, time_emb1):
        #diffusion_embedding = self.time_embedding(diffusion_timestep)
        
        x = self.in_project(x, cond, time_emb1)
        n = len(self.denoising)
        for i in range(n):
            x = self.denoising[i](x, cond, time_emb1)
            
        x = self.out_project(x, cond, time_emb1)
            
        return x



class Diffusion(nn.Module):
    def __init__(self, n_layers, diffusion_time_embedding_dim,
                 latent_dim,diff_embed_dim,vital_sign_dim,latent_dim_auto,
                 hidden_auto,n_times=1000, beta_minmax=[1e-4, 2e-2],
                 device='cuda',cls_emb_dim=768,
                 ts_cond=True,L=1.0,d_cls=0.,sentence_del=0.,
                 lambda1=1.0, lambda2=1.0,
                 lambda0=0.5,tp_embed_dim=128,tp_layer=2):
    
        super(Diffusion, self).__init__()

        cls_emb_dim = 768 # 768 for gatortron otherwise 768
        self.n_times = n_times
        self.device = device
        self.total_times = n_times + 1
        self.latent_dim = latent_dim
        self.x_ts_dim = vital_sign_dim
        self.cls_emb_dim = cls_emb_dim
        self.diff_embed_dim = diff_embed_dim
        self.cls_criterion = nn.BCELoss()
        self.num_sample = 10
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ts_cond = ts_cond
        latent_dim_auto = cls_emb_dim
        self.latent_dim_auto = cls_emb_dim
        self.hidden_auto = hidden_auto
        self.d_cls = d_cls #4 # average, consider 1 std; [-1,1]
        self.L = L
        self.sentence_del = sentence_del #2.5

        self.model = Denoiser(latent_dim_auto, latent_dim,diff_embed_dim, n_layers,
                              diffusion_time_embedding_dim=diffusion_time_embedding_dim,
                              n_times=n_times, ts_cond=ts_cond)

        #self.autoencoder = Autoencoder(cls_emb_dim, hidden_auto,
        #                               latent_dim_auto, n_layers+2) 


        self.t_prime_predictor = t_prime_predictor(cls_emb_dim, self.total_times,
                                                   tp_embed_dim, tp_layer)

        
        
        # define linear variance schedule(betas)
        beta_1, beta_T = beta_minmax
        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times) # follows DDPM paper
        self.betas = torch.cat((torch.tensor([0.]), self.betas),dim=0).to(device)
        self.sqrt_betas = torch.sqrt(self.betas)
                                     
        # define alpha for forward diffusion kernel
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.one_over_sqrt_alphas = 1 / torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alphas = torch.sqrt(1-self.alphas)
        # matrix, A[t', t] = 0, if t' = t, \bar{alpha}_{t':t}
        #self.alpha_bar_trun = gen_alpha_trunc(self.alphas).to(device)
        
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-self.alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sinusoidalPosEmb = SinusoidalPosEmb(diffusion_time_embedding_dim)
        self.inference_time_noise_coeffs = self.Cal_inference_time_noise_coeff()
        self.time_embeddings = self.Create_time_emb()
        self.normalized_schedule = (torch.arange(self.total_times) / \
                                    self.total_times).to(self.device)+1e-04
        ### New normalized_schedule ####
        #normalized_schedule = torch.arange(self.total_times).to(self.device) + 0.1
        #self.normalized_schedule = normalized_schedule / torch.sum(normalized_schedule).item()
        #################################
        self.post_sqrt_var = self.Cal_post_sqrt_var()
        # avoid the first element being zero
        
        ## parameters for ELBO_0_tp
        self.t_in_0_tp = None
        self.predicted_noise_mean = None
        self.predicted_noise_cov = None
        self.t_p = None # sample once in training
        self.starting_t_p = None # for inference
        self.bs = None
        self.seq_len = None

    def Cal_post_sqrt_var(self):
        alpha_bar_t_minus_1 = self.alpha_bars[:-1]
        var = self.betas[1:] * (1 - alpha_bar_t_minus_1) / (1 - self.alpha_bars[1:])
        var = torch.cat((torch.tensor([0.]).to(self.device), var),dim=0)
        return torch.sqrt(var) # first two elements = 0 corresponding to t=0,1

    def Cal_inference_time_noise_coeff(self):
        results = []
        for t in range(self.total_times):
            
            alpha = self.alphas[t]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
            results.append((1-alpha)/sqrt_one_minus_alpha_bar)
        return torch.tensor(results).to(self.device)

    def Create_time_emb(self):
        emb = []
        for i in range(self.total_times):
            emb.append(self.sinusoidalPosEmb(torch.tensor([i]).float()).squeeze(0))
        return torch.stack(emb,0).to(self.device)

    
    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5
    
    def make_noisy(self, x_t_prime, t_p, t, sample=None):
        # x_t_prime: [bs, dim], t_p:[bs, 1], t:[bs, 1]
        # perturb x_t' into x_t
        # bs * seq_len = len(t_p.flatten()) = len(t.flatten())
        if sample is None:
            epsilon = torch.randn_like(x_t_prime).to(self.device)
            # alpha bar t'+1:t
        else:
            bs, dx = x_t_prime.shape
            x_t_prime = x_t_prime.repeat(1, sample).view(bs*sample,dx)
            epsilon = torch.randn_like(x_t_prime).to(self.device)
            t_p = t_p.repeat(1,sample).view(bs*sample,1)
            t = t.repeat(1,sample).view(bs*sample,1)
        t_p = t_p.long()
        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_tp1_t) * x_t_prime + sqrt(1-sqrt_one_minus_alpha_bar_tp1_t) * epsilon
        sqrt_alpha_bar_tp1_t = self.sqrt_alpha_bars[t.flatten()] / self.sqrt_alpha_bars[t_p.flatten()]
        alpha_bar_tp1_t = self.alpha_bars[t.flatten()] / self.alpha_bars[t_p.flatten()]
        sqrt_one_minus_alpha_bar_tp1_t = torch.sqrt(1 - alpha_bar_tp1_t)
        noisy_sample = sqrt_alpha_bar_tp1_t.unsqueeze(-1) * x_t_prime + \
                       epsilon * sqrt_one_minus_alpha_bar_tp1_t.unsqueeze(-1)
    
        return noisy_sample, epsilon



    def elbo_diff(self, z_t_prime, t_p, cond):
        # z_t_prime: [bs, dim], t_p: [bs, 1]
        # cond: [bs, emb_dim]
        bs = z_t_prime.shape[0]
        # (1) randomly choose diffusion time-step
        t = torch.stack([torch.randint(low=t_p[i][0],high=self.total_times,
                                size=[1]) for i in range(bs)]).to(self.device)
        
        # (2) forward diffusion process: perturb x_t_prime with fixed
        # variance schedule
        x_t, epsilon = self.make_noisy(z_t_prime, t_p, t)
        t_emb = self.time_embeddings[t.flatten()]
        
        # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
        pred_epsilon = self.model(x_t, cond, t_emb)
        mask = 1 - (t_p == t).to(torch.int) # if t' = t, mask entry = 0
        loss = mask * (epsilon - pred_epsilon) ** 2
        return loss



    def get_encode_tp(self, x_cls):
        #z = self.autoencoder.encode(x_cls)
        z = x_cls
        tp_densities = self.t_prime_predictor(x_cls)
        return z, tp_densities


    def loss1(self, inputs):
        x_cls,x_cls_inc, t_txt, mask_txt1,\
             x_ts, t_ts,cond,x0,tp_densities = inputs
        return self.loss(x_cls,x_cls_inc, t_txt, mask_txt1,
             x_ts, t_ts,cond,x0,tp_densities)




    def upper_bound(self, x_cls, x_cls_inc, mask_txt1,
                    x0,tp_densities):
        cond = None
        bs,seq_len,dx = x_cls.shape
        self.bs = bs
        self.seq_len = seq_len
        bs_seq_len = bs * seq_len
        mask_txt = mask_txt1.mean(-1)
        mask_txt = mask_txt.view(bs_seq_len,1)
        x_cls = x_cls.view(bs_seq_len, dx)

        x_cls_inc = x_cls_inc.view(bs_seq_len, dx)
        tp_densities = tp_densities.view(bs_seq_len, self.total_times)
        z_cor, tp_densities1 = self.get_encode_tp(x_cls_inc)
        del x_cls_inc
        expected_tp = (tp_densities * self.normalized_schedule).sum(-1).unsqueeze(-1)


        L_complete = self.loss_complete(x_cls, x0.view(bs*seq_len,self.cls_emb_dim),
                                        tp_densities, tp_densities1,
                                        z_cor, cond, mask_txt, bs, seq_len, expected_tp)
        return L_complete
    
    
    def loss(self, x_cls,x_cls_inc, mask_txt1,
             x0,tp_densities,cond=None,pretrain=False):
        # x_cls, report cls embeddings with padded data, shape=[bs,seq_len, 768]
        # x_cls_inc, incomplete report cls embeddings with padded data, shape=[bs, seq_len, 768]
        # t_txt: timestamps, shape=[bs, seq_len]
        # mask_txt1: 1 non-padded, 0 --- padded report, shape=[bs, seq_len,768]
        # x_ts: time series with padded data, shape=[bs, seq_len2, dx]
        # t_ts: timestamps, shape=[bs, seq_len2]
        # cond: [bs, seq_len, latent_dim], embeddings of ts queried by txt time
        # tp_densities: [bs,seq_len,total_times], probability of t' for
        # orginal txt embedding

        
        bs,seq_len,dx = x_cls.shape
        self.bs = bs
        self.seq_len = seq_len
        bs_seq_len = bs * seq_len

        mask_txt = mask_txt1.mean(-1)
        mask_txt = mask_txt.view(bs_seq_len,1)

        x_cls = x_cls.view(bs_seq_len, dx)
        # first dimension [bs1_t1,bs1_t2,bs1_t3,bs2_t1,bs2_t2,...]
        # x_cls_inc = x_cls_cor
        x_cls_inc = x_cls_inc.view(bs_seq_len, dx)
        
        
        ## calc contrastive loss between complete and incomplete reports
        #z, tp_densities = self.get_encode_tp(x_cls)
        tp_densities = tp_densities.view(bs_seq_len, self.total_times)
        #z = x_cls
        z_cor, tp_densities1 = self.get_encode_tp(x_cls_inc)

        if pretrain:
            t0_given = tp_densities[:,0].unsqueeze(-1)
            t0_inc = tp_densities1[:,0].unsqueeze(-1)
            expected_tp = (tp_densities * self.normalized_schedule).sum(-1).unsqueeze(-1)
            expected_tp1 = (tp_densities1 * self.normalized_schedule).sum(-1).unsqueeze(-1)
            L_contrast = self.loss_contrast(expected_tp, expected_tp1, t0_given,
                                        t0_inc, mask_txt)
            return L_contrast
        del x_cls_inc
        tp_densities = tp_densities.detach()
        tp_densities1 = tp_densities1.detach()
        expected_tp = (tp_densities * self.normalized_schedule).sum(-1).unsqueeze(-1)
        
        ## calc complete loss
        t_p = torch.randint(low=0, high=self.total_times,
                            size=[bs, seq_len]).to(self.device).view(bs_seq_len,1)
        self.t_p = t_p
        self.init_params_ELBO_0_tp(bs_seq_len)
        #expected_z0 = self.infer_z0(z.view(bs,seq_len,self.latent_dim_auto),
        #                            tp_densities.view(bs,seq_len,self.total_times),
        #                            cond,store=True)
        L_complete = self.loss_complete(x_cls, x0.view(bs*seq_len,self.cls_emb_dim),
                                        tp_densities, tp_densities1,
                                        z_cor, cond, mask_txt, bs, seq_len, expected_tp)
        del z_cor
        
        
        #L_contrast = (mask_txt * (expected_tp - expected_tp1)).mean()

        ## calc inputs
        # the range of t' and t is [0,1,2,...,T]
        # tp_densities: [bs, seq_len, total_times]
        
        
        
        p_tp_given_z = tp_densities.gather(-1, t_p)
        z_t_prime = self.make_noisy(x_cls,
                                    torch.zeros(bs_seq_len, 1).int().to(self.device),
                                    t_p)[0]


        

        ## calc ELBO
        ELBO_tT = self.elbo_diff(z_t_prime, t_p,
                                 cond)
        
        ELBO_0t = self.elbo_infer()
        ELBO_tT_mask = t_p != self.n_times
        ELBO_0t_mask = t_p != 0
        L_ELBO = (mask_txt * p_tp_given_z *\
                 (ELBO_tT_mask * ELBO_tT * (1-self.lambda0) + \
                  ELBO_0t_mask * ELBO_0t * self.lambda0)\
                  / (ELBO_tT_mask + ELBO_0t_mask)).mean()
        
        
        

        return L_complete +  2.0 * L_ELBO


    def loss_contrast(self,tp_txt, tp_cor, t0_given, t0_inc, mask_txt):
        # tp_txt, tp_cor, t0_given, t0_inc dimension: [bs_seq_len, 1]
        # mask_txt: [bs, seq_len]
        # self.d_cls = \hat{y}
        # self.lambda2 = expected t'
        bs, seq_len = mask_txt.shape
        bs_seq_len = bs * seq_len
        bs_seq_len_square = bs_seq_len ** 2
        mask_txt1 = mask_txt.view(bs_seq_len, 1).repeat(1,bs_seq_len).\
                    view(bs_seq_len_square,1)
        mask_txt2 = mask_txt.view(bs_seq_len, 1).repeat(bs_seq_len,1).\
                    view(bs_seq_len_square,1)

        tp_txt1 = tp_txt.repeat(1,bs_seq_len).view(bs_seq_len_square, 1)
        tp_cor1 = tp_cor.repeat(bs_seq_len,1)
        numerator = ((self.lambda1 * torch.log(tp_cor1)) + \
                      (1 - self.lambda1) * torch.log(1 - torch.log(tp_txt1)))
        denominator = (mask_txt2 * numerator)\
                  .view(bs_seq_len,bs_seq_len).sum(1).unsqueeze(-1)\
                      .repeat(1,bs_seq_len).view(bs_seq_len_square,1)
        ones_idx = torch.linspace(0,bs_seq_len_square-1,bs_seq_len).int()
        w = numerator / denominator
        w[ones_idx] = 1.

        loss = (mask_txt1 * mask_txt2 * w * (tp_txt1 - tp_cor1)).mean()


        return loss
        
        
    def rank_mask(self, x, percent):
        # x: [bs, 1]
        n = x.shape[0]
        sort_x = torch.sort(x.flatten())[0]
        idx = int(n * (1.0-percent))
        lowest = sort_x[idx]
        mask = x >= lowest
        return mask.float().to(self.device)

        
        
        

    def loss_complete(self, x_cls,x0, tp_densities_txt,tp_densities_cor,
                      z_cor, cond, mask_txt,bs, seq_len, expected_tp):
        # x_cls: [bs*seq_len, 768]
        # x0: [bs*seq_len, 768]
        # tp_densities_txt: [bs*seq_len, T+1]
        # z_cor: [bs*seq_len, dz]
        # cond: [bs, seq_len, self.latent_dim]
        # mask_txt: [bs*seq_len, 1]

        p_tp_0 = tp_densities_txt[:,0].unsqueeze(-1)
        
        
        expected_x0_cor = self.infer_z0(z_cor.view(bs,seq_len,self.latent_dim_auto),
                                    tp_densities_cor.view(bs,seq_len,self.total_times),
                                    cond).view(bs*seq_len, self.cls_emb_dim)
        percentile_complete = self.lambda2
        tp_0_mask = self.rank_mask(p_tp_0, percentile_complete)
        loss1 = mask_txt * \
                ((tp_0_mask * (x_cls - expected_x0_cor)**2))

        
        del x_cls # save memory
        loss2 = mask_txt * ((x0
                             - expected_x0_cor)**2)
        

        return (loss1+loss2).mean()
        
        
    
    
    def denoise_at_t(self, x_t, cond, t, store=False):
        # t: int, minimum=1
        # x_t: [bs, dim]
        B = x_t.shape[0]

        z = torch.randn_like(x_t).to(self.device)
        t = torch.tensor(t).repeat(B).unsqueeze(-1).to(self.device)
        
        t_emb = self.time_embeddings[t.flatten()]
        epsilon_pred = self.model(x_t, cond, t_emb)
        one_over_sqrt_alpha = self.one_over_sqrt_alphas[t]
        noise_coeff = self.inference_time_noise_coeffs[t]
        if self.t_in_0_tp is not None and store:
            self.store_noise(epsilon_pred, t)
        # denoise at time t, utilizing predicted noise
        post_sqrt_var = self.post_sqrt_var[t]
        x_t_minus_1 = one_over_sqrt_alpha * (x_t - noise_coeff*epsilon_pred)\
                      + post_sqrt_var*z
        # t==1, then post_sqrt_var = 0
        #x_t_minus_1 = one_over_sqrt_alpha * (x_t - noise_coeff*epsilon_pred)
        return x_t_minus_1

                
    def infer(self, x_t_prime, cond, t_p, store=False):
        # t_p: int
        x_t = x_t_prime
        if t_p == 0:
            return x_t_prime
        else:
            if store:
                self.starting_t_p = t_p
            for i in range(t_p, 0, -1):
                x_t = self.denoise_at_t(x_t, cond, i, store=store)
            return x_t

    def infer_z0(self, z, tp_densities, cond, store=False):
        # z: [bs, seq_len, dz]
        # tp_densities: [bs, seq_len, self.total_times]
        # cond: [bs, seq_len, self.latent_dim]
        # store: whether store pre_noise and params
        sample = self.num_sample
        bs, seq_len,dz = z.shape
        bs1 = bs*seq_len
        x0 = torch.zeros(bs1, self.cls_emb_dim).to(self.device)
        z = z.view(bs1, dz)
        tp_densities = tp_densities.view(bs1, self.total_times)
        t_p = torch.zeros(bs1, 1).int().to(self.device)
        if cond is not None:
            cond = cond.repeat(1,1,sample).view(bs*seq_len*sample,
                                                self.latent_dim)
        for i in range(self.n_times, -1, -1):
            t = torch.tensor([[i]]).repeat(bs1,1).to(self.device)
            zt,_ = self.make_noisy(z, t_p, t, sample=sample)
            x0_temp = self.infer(zt, cond, i, store=store)\
                      .view(bs1, sample, self.latent_dim_auto).mean(1)
            x0 += tp_densities[:,i].unsqueeze(-1) * x0_temp
        return x0.view(bs, seq_len,self.cls_emb_dim)


    def forward(self, z):
        tp_densities = self.t_prime_predictor(z)
        expected_tp = (tp_densities * self.normalized_schedule).sum(-1)
        return tp_densities[:,:,0].flatten().tolist(),\
               expected_tp.flatten().tolist()
    

    def init_params_ELBO_0_tp(self, bs_seq_len):
        # t_p: [bs_seq_len,1]
        # tp_densities: [bs_seq_len, T+1]
        # z_t_prime: [bs_seq_len, dim]
        
        # sample t for inference
        bs = bs_seq_len
        # random select t between t' and 0
        self.t_in_0_tp = torch.stack([torch.randint(low=0,high=self.t_p[i][0]+1,
                                size=[1]) for i in range(bs)]).to(self.device)

        # always set t 0 zero to consider denoising all the way to 0
        #self.t_in_0_tp = torch.zeros(bs,1).int().to(self.device)

        dim = self.latent_dim_auto
        self.predicted_noise_mean = torch.zeros(1).to(self.device)
        self.predicted_noise_cov = torch.zeros(1).to(self.device)
        # repeat for sample
        #bs_seq_len_sample = bs_seq_len * self.num_sample
        #self.t_in_0_tp = self.t_in_0_tp.repeat(1, self.num_sample)\
        #                 .view(bs_seq_len_sample, 1)
        #self.t_p = self.t_p.repeat(1, self.num_sample)\
        #      .view(len(self.t_p)*self.num_sample, 1)
        #self.predicted_noise = self.predicted_noise.repeat(1, self.num_sample)\
        #                       .view(bs_seq_len_sample, dim)



    def reset_params_ELBO_0_tp(self):
        self.t_in_0_tp = None
        self.predicted_noise_mean = None
        self.predicted_noise_cov = None
        self.t_p = None
        self.starting_t_p = None
        self.bs = None
        self.seq_len = None

    def store_noise(self, epsilon_pred, t):
        # epsilon_pred: [bs_seq_len_sample,dim]
        # t: [bs_seq_len_sample,1], 0 < t <= tp, if tp=0, this
        # function would not be called
        # self.t_p minimum = 1

        mask = self.t_p == self.starting_t_p
        mask1 = t.view(self.bs*self.seq_len, self.num_sample, 1)[:,0,:]\
                == self.t_in_0_tp
        mask = (mask * mask1).float().to(self.device)
        predicted_noise = epsilon_pred.\
                                 view(self.bs*self.seq_len,
                                      self.num_sample,self.latent_dim_auto)
                                    #[bs_seq_len,  num_sample, 768]
        self.predicted_noise_mean = torch.cat((self.predicted_noise_mean,
                                               (predicted_noise ** 2).mean(2).mean(1)*mask), 0)
        self.predicted_noise_cov = torch.cat((self.predicted_noise_cov,
                                              ((torch.diag(torch.cov(predicted_noise.view(-1, self.latent_dim_auto))) - 1.0) ** 2)\
                                              .view(-1, self.num_sample).mean(1) *  mask), 0)
        

        
    def elbo_infer(self):
        bs = self.t_p.shape[0]
        #num_denoise_steps = (self.t_p - self.t_in_0_tp) + 1
        #noise = self.predicted_noise / num_denoise_steps 
        #noise = noise.view(int(bs/self.num_sample), self.num_sample, dim)\
        #        .mean(1)
        #return noise
        return self.predicted_noise_mean.mean() + self.predicted_noise_cov.mean()
