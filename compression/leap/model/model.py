import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
import math
import random
from model.backbone import build_backbone, BackboneOutBlock
from model.encoder import CrossViewEncoder
from model.neck import PETransformer
from model.single_stream_transformer import SingleStream
from model.render_module import RenderModule


def gelu(x):
  return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

class LEAP(nn.Module):
    def __init__(self, config) -> None:
        super(LEAP, self).__init__()
        self.config = config

        # input and output size
        self.input_size = config.dataset.img_size
        self.render_size = config.dataset.img_size_render   # use smaller render size for saving memory
        self.linear_c  = nn.ModuleList([
            nn.Linear(4, config.model.backbone_out_dim, bias=False),\
            nn.BatchNorm2d(config.model.backbone_out_dim), \
            nn.GELU(),\
            nn.Linear(config.model.backbone_out_dim, config.model.backbone_out_dim, bias=False),\
            nn.BatchNorm2d(config.model.backbone_out_dim), \
            nn.GELU(),\
            nn.Linear(config.model.backbone_out_dim, config.model.backbone_out_dim, bias=False),\
            nn.BatchNorm2d(config.model.backbone_out_dim), \
            nn.GELU(),\
            nn.Linear(config.model.backbone_out_dim, config.model.backbone_out_dim, bias=False),\
            nn.BatchNorm2d(config.model.backbone_out_dim), \
            nn.GELU(),\
            nn.Linear(config.model.backbone_out_dim, config.model.backbone_out_dim, bias=False),\
            nn.BatchNorm2d(config.model.backbone_out_dim), \
            nn.GELU()])
        
        embedding_stdev = (1. / math.sqrt(config.model.backbone_out_dim))
        self.view_encoding_r = nn.parameter.Parameter(torch.rand(config.model.backbone_out_dim) * embedding_stdev)
        self.view_encoding_s = nn.parameter.Parameter(torch.rand(config.model.backbone_out_dim) * embedding_stdev)

        # build backbone
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name
        self.backbone_out_dim = config.model.backbone_out_dim

        self.backbone_out = BackboneOutBlock(in_dim=self.backbone_dim, out_dim=self.backbone_out_dim)
        self.feat_res = int(self.input_size // self.down_rate)

        # # build cross-view feature encoder
        # self.encoder = CrossViewEncoder(config, in_dim=self.backbone_out_dim, in_res=self.feat_res)

        # build p.e. transformer
        # if config.model.use_neck:
        #     self.neck = PETransformer(config, in_dim=self.backbone_out_dim, in_res=self.feat_res)
        
        # build 2D-3D lifting
        self.singlestream = SingleStream(config)

        # build 3D-2D render module
        self.render_module = RenderModule(config, feat_res=self.feat_res)


    def extract_feature(self, x, c, return_h_w=False):
        if self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            #out = self.backbone.get_intermediate_layers(x, c, n=1)[0]
            out = self.backbone(x, c).feature_maps[0] ###(b, hidden, h, w,)
            
            #h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            #dim = out.shape[1]
            #out = out.reshape(b, h, w, dim).permute(0,3,1,2)
        else:
            raise NotImplementedError('unknown image backbone')
        return out


    def forward(self, sample, device, return_neural_volume=False, render_depth=False):
        '''
        imgs in shape [b,t,C,H,W]
        '''
        t_input = self.config.dataset.num_frame
        imgs = sample['images'].to(device)[:,:t_input]
        c_pos = self.linear_c(sample['camera_intri'].to(device)) ###hidden_dim
        
        bsz = c_pos.size(0)
        view_encoding = torch.cat([self.view_encoding_r.unsqueeze(0), self.view_encoding_s.repeat(t_input-1, 1)], dim=0).repeat(bsz,1,1)
        c_pos += view_encoding
        b, t = imgs.shape[:2]
        assert t == t_input and bsz == b
        # 2D per-view feature extraction
        imgs = rearrange(imgs, 'b t c h w -> (b t) c h w')
        c_pos = rearrange(c_pos, 'b t f -> (b t) f')
        if self.config.model.backbone_fix:
            with torch.no_grad():
                features = self.extract_feature(imgs, c_pos)                           # [b*t,c=768,h,w]
        else:
            features = self.extract_feature(imgs, c_pos)
        
        features = self.backbone_out(features)
        features = rearrange(features, '(b t) c h w -> b t c h w', b=b, t=t)    # [b,t,c,h,w]

        # cross-view feature refinement
        x_img, x_3d = self.singlestream(features)
                                           
        # 2D-3D lifting
        ######################################
        ######################################
        ## stop here stop  here
        ######################################
        features_3d_raw, features_3d = self.lifting(features, pe2d)             # [b,c=768,D=16,H,W], [b,c=128,D=64,H,W]

        # rendering
        results = self.render_module(features_3d, sample, return_neural_volume, render_depth, features_3d_raw)

        # return 2D features if necessary (legacy, not used)
        if self.config.model.render_feat_raw and self.training:
            results['features_2d'] = rearrange(pe2d, 'b t c h w -> (b t) c h w')
        
        return results






