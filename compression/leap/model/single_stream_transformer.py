import torch.nn as nn
import torch
import torch.nn.functional as F
from model.base_module.cross_attention import TransformerCrossAttnLayer
from model.base_module.TransformerDecoder import TransformerDecoderLayerPermute
from model.base_module.flash_attention.transformer import FlashTxDecoderLayer, FlashCrossAttnLayer
from einops import rearrange
import math
from utils.pe_utils import positionalencoding3d

def lifting_make_conv3d_layers(config, in_dim, out_dim):
    if config.model.lifting_use_conv3d:
        conv3ds = [nn.Sequential(nn.Conv3d(in_dim, in_dim, 3, padding=1),
                             nn.BatchNorm3d(in_dim),
                             nn.LeakyReLU(inplace=True),) for _ in range(num_layers)]
    else:
        conv3ds = [nn.Identity() for _ in range(num_layers)]
    conv3ds = nn.ModuleList(conv3ds)
    return conv3ds

class SingleStream(nn.Module):
    def __init__(self, config) -> None:
        super(SingleStream, self).__init__()
        self.config = config
        in_dim = config.model.hidden_size
        embedding_stdev = (1. / math.sqrt(in_dim))
        self.latent_res = config.model.latent_res
        self.latent_emb = nn.parameter.Parameter(
                            (torch.rand(self.latent_res * self.latent_res * self.latent_res, config.model.backbone_out_dim) * embedding_stdev))
        self.proj = nn.Linear(config.model.backbone_out_dim, in_dim, bias=False),
        self.transformer = nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=config.model.heads, dim_feedforward=in_dim, 
                                                        dropout=0.0, activation='gelu', batch_first=True, norm_first=True) \
                                                            for _ in range(config.model.layers)])

        self.deconv3d = lifting_make_conv3d_layers(config, in_dim)

    def forward(self, x):
        '''
        x: 2D features in shape [b,t,c,h,w]
        pe2d: 2D p.e. in shape [b,t,c,h,w]
        '''
        b,t,c,h,w = x.shape
        device = x.device

        if self.latent_emb.device != device:
            self.latent_emb = self.latent_emb.to(device)
            
        x = rearrange(x, 'b t c h w -> b (t h w) c') ##(b, )
        x = torch.cat([self.latent_emb.repeat(b,1,1), x], dim=1)
        
        x = self.transformer(self.proj(x))
        ######
        x_img = x[:, self.latent_res**3:]
        x_3d = x[:, :self.latent_res**3]
        x_3d = self.deconv3d(x_3d)
        return x_img, x_3d


