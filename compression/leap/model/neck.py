import torch.nn as nn
import torch
import torch.nn.functional as F
from model.base_module.flash_attention.transformer import FlashSelfAttnLayer
from einops import rearrange
import math


class PETransformer(nn.Module):
    def __init__(self, config, in_dim, in_res) -> None:
        super(PETransformer, self).__init__()

        self.config = config

        embedding_stdev = (1. / math.sqrt(in_dim))
        self.pe_canonical = nn.parameter.Parameter(torch.rand(in_dim, in_res, in_res) * embedding_stdev)

        self.pe_transformer = neck_make_transformer_layers(config, in_dim)

        if config.model.pe_with_spatial_pe:
            self.pe_spatial = nn.parameter.Parameter(torch.rand(10, in_dim, in_res, in_res) * 1e-4)

        if config.model.neck_scale == 'channel':
            self.scale = None
        else:
            self.scale = float(config.model.neck_scale.split('_')[-1])


    def transform_pe(self, x_q, x_k, pe):
        '''
        x_q: in [b,h*w,c]
        x_k: in [b,c,h*w]
        pe: in [b,c,h,w]
        '''
        b,c,h,w = pe.shape
        pe = rearrange(pe, 'b c h w -> b (h w) c')
        # scale = c ** -0.5 if self.scale is None else self.scale
        # attn = torch.matmul(x_q, x_k) / scale   # [b,h*w,c] @ [b,c,h*w] -> [b,h*w,h*w]

        scale = c ** 0.5 if self.scale is None else self.scale
        attn = torch.matmul(x_q, x_k) / scale
        attn = F.softmax(attn, dim=-1)
        pe_q = torch.matmul(attn, pe)           # [b,h*w,h*w] @ [b,h*w,c] -> [b,h*w,c]
        pe_q = rearrange(pe_q, 'b (h w) c -> b c h w', h=h,w=w)
        return pe_q


    def forward(self, x):
        '''
        x in shape [b,t,c,h,w]
        '''
        b,t,c,h,w = x.shape

        pe_canonical = self.pe_canonical.unsqueeze(0).repeat(b,1,1,1)       # [b,c,h,w]

        # get per-view p.e.
        x_canonical = x[:,0]                                                # [b,c,h,w]
        x_canonical = rearrange(x_canonical, 'b c h w -> b c (h w)')        # [b,c,h*w]
        pe_noncanonical = []
        for i in range(t-1):
            x_cur = x[:,i+1]                                                # [b,c,h,w]
            x_cur = rearrange(x_cur, 'b c h w -> b (h w) c')                # [b,h*w,c]
            pe_cur = self.transform_pe(x_cur, x_canonical, pe_canonical)    # [b,c,h,w]
            pe_noncanonical.append(pe_cur)
        pe_noncanonical = torch.stack(pe_noncanonical, dim=1)               # [b,t-1,c,h,w]

        pe = torch.cat([pe_canonical.unsqueeze(1), pe_noncanonical], dim=1) # [b,t,c,h,w]

        # refine p.e.
        pe = rearrange(pe, 'b t c h w -> b (t h w) c')

        # add p.e. with spatial p.e.
        if self.config.model.pe_with_spatial_pe:
            pe_spatial = self.pe_spatial[:t].unsqueeze(0)   #[1,t,c,h,w]
            pe_spatial = rearrange(pe_spatial, 'b t c h w -> b (t h w) c')
            pe += pe_spatial

        pe = self.pe_transformer(pe)
        pe = rearrange(pe, 'b (t h w) c -> b t c h w', t=t, h=h, w=w)
        return pe
        # x = x + pe
        # return x


def neck_make_transformer_layers(config, in_dim):
    pe_transformer = []
    num_layers = config.model.neck_layers
    mlp_ratio = 4.0
    norm_first = config.model.norm_first

    if not config.model.use_flash_attn:
        latent_dim = int(mlp_ratio * in_dim)
        pe_transformer = nn.Sequential(*[
            torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim,
                                             dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            for _ in range(num_layers)    
        ])
    else:
        pe_transformer = nn.Sequential(*[
            FlashSelfAttnLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=norm_first)
            for _ in range(num_layers)    
        ])
    return pe_transformer