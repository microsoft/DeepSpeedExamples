import torch.nn as nn
import torch
import torch.nn.functional as F
from model.base_module.cross_attention import TransformerCrossAttnLayer
from model.base_module.flash_attention.transformer import FlashSelfAttnLayer, FlashCrossAttnLayer
from einops import rearrange
import math

class CrossViewEncoder(nn.Module):
    def __init__(self, config, in_dim, in_res) -> None:
        super(CrossViewEncoder, self).__init__()
        self.config = config

        self.transformers_cross, self.transformers_self = make_encoder_transformer_layers(config, in_dim)
        #self.transformer_self = make_encoder_transformer_layers_self_only(config, in_dim)

        # 2d positional embedding
        embedding_stdev = (1. / math.sqrt(in_dim))
        self.pixel_emb = nn.parameter.Parameter(torch.rand(1, in_dim, in_res, in_res) * embedding_stdev)
        # camera ID embedding (see scene representation transformer)
        self.cam_emb = nn.parameter.Parameter(torch.rand(10, in_dim, 1, 1) * embedding_stdev)
        
        
    def forward(self, x):
        '''
        x in shape [b,t,c,h,w]
        '''
        b,t,c,h,w = x.shape

        # The positional embeddings and camera ID embeddings are not useful and degenerates the performance a bit,
        # While I am too lazy to train a now model without it.
        # You can train a model without these embeddings to verify the point.
        # add 2D positional embedding (same for each image)
        w_h_diff_half = (w - h) // 2
        x = x + self.pixel_emb.unsqueeze(0)[:,:,:,int(w_h_diff_half):int(w-w_h_diff_half),:w]                # [b,t,c,h,w]
        # add camera ID embedding (different between images, same for all pixels in one image)
        cam_emb = self.cam_emb[:t].repeat(1,1,h,w).unsqueeze(0)
        x = x + cam_emb                                                 # [b,t,c,h,w]

        # get canonical view
        x_canonical = x[:, 0]                                           # [b,c,h,w]
        x = x[:, 1:]                                                    # [b,t-1,c,h,w]
        x_canonical = rearrange(x_canonical, 'b c h w -> b (h w) c')
        x = rearrange(x, 'b t c h w -> b (t h w) c')

        # get through transformer encoder
        for (cross_attn, self_attn) in zip(self.transformers_cross, self.transformers_self):   # requires [b,n,c] inputs
            # cross-attention between canonical-other frames
            x = cross_attn(x, memory=x_canonical)       # [b,(t-1)*h*w,c]
            # concat all frame features
            x = rearrange(x, 'b (t h w) c -> b t c h w', t=t-1, h=h, w=w)
            x_canonical = rearrange(x_canonical, 'b (t h w) c -> b t c h w', t=1, h=h, w=w)
            x = torch.cat([x_canonical, x], dim=1)          # [b,t,c,h,w]
            x = rearrange(x, 'b t c h w -> b (t h w) c')    # [b,n=t*h*w,c]
            # self-attention refinement for all frames
            x = self_attn(x)
            # split the canonical and other frame features
            x = rearrange(x, 'b (t h w) c -> b t c h w', t=t, h=h, w=w)
            x_canonical = x[:, 0]
            x = x[:, 1:]
            x_canonical = rearrange(x_canonical, 'b c h w -> b (h w) c')
            x = rearrange(x, 'b t c h w -> b (t h w) c')

        x_canonical = rearrange(x_canonical, 'b (t h w) c -> b t c h w', t=1, h=h, w=w)
        x = rearrange(x, 'b (t h w) c -> b t c h w', t=t-1, h=h, w=w)
        x = torch.cat([x_canonical, x], dim=1)          # [b,t,c,h,w]

        return x


def make_encoder_transformer_layers(config, in_dim):
    transformers_cross, transformers_self = [], []
    num_layers = config.model.encoder_layers
    mlp_ratio = 4.0
    norm_first = config.model.norm_first

    if not config.model.use_flash_attn:
        latent_dim = int(mlp_ratio * in_dim)
        for _ in range(num_layers):
            cross_attn = TransformerCrossAttnLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim, 
                                                   dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            self_attn = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=8, dim_feedforward=latent_dim,
                                                         dropout=0.0, activation='gelu', batch_first=True, norm_first=norm_first)
            transformers_cross.append(cross_attn)
            transformers_self.append(self_attn)
    else:
        for _ in range(num_layers):
            cross_attn = FlashCrossAttnLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=norm_first)
            self_attn = FlashSelfAttnLayer(d_model=in_dim, n_head=12, mlp_ratio=mlp_ratio, norm_first=norm_first)
            transformers_cross.append(cross_attn)
            transformers_self.append(self_attn)

    transformers_cross = nn.ModuleList(transformers_cross)
    transformers_self = nn.ModuleList(transformers_self)
    return transformers_cross, transformers_self
