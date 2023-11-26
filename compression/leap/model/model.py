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
from model.lifting import lifting
from model.render_module import RenderModule


class LEAP(nn.Module):
    def __init__(self, config) -> None:
        super(LEAP, self).__init__()
        self.config = config

        # input and output size
        self.input_size = config.dataset.img_size
        self.render_size = config.dataset.img_size_render   # use smaller render size for saving memory
        
        # build backbone
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name
        self.backbone_out_dim = config.model.backbone_out_dim
        self.backbone_out = BackboneOutBlock(in_dim=self.backbone_dim, out_dim=self.backbone_out_dim)
        self.feat_res = int(self.input_size // self.down_rate)

        # build cross-view feature encoder
        self.encoder = CrossViewEncoder(config, in_dim=self.backbone_out_dim, in_res=self.feat_res)

        # build p.e. transformer
        if config.model.use_neck:
            self.neck = PETransformer(config, in_dim=self.backbone_out_dim, in_res=self.feat_res)
        
        # build 2D-3D lifting
        self.lifting = lifting(config, self.backbone_out_dim)

        # build 3D-2D render module
        self.render_module = RenderModule(config, feat_res=self.feat_res)


    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
        else:
            raise NotImplementedError('unknown image backbone')
        return out


    def forward(self, sample, device, return_neural_volume=False, render_depth=False):
        '''
        imgs in shape [b,t,C,H,W]
        '''
        t_input = self.config.dataset.num_frame
        imgs = sample['images'].to(device)[:,:t_input]
        b,t = imgs.shape[:2]
        
        # 2D per-view feature extraction
        imgs = rearrange(imgs, 'b t c h w -> (b t) c h w')
        if self.config.model.backbone_fix:
            with torch.no_grad():
                features = self.extract_feature(imgs)                           # [b*t,c=768,h,w]
        else:
            features = self.extract_feature(imgs)
        features = self.backbone_out(features)
        features = rearrange(features, '(b t) c h w -> b t c h w', b=b, t=t)    # [b,t,c,h,w]

        # cross-view feature refinement
        features = self.encoder(features)                                       # [b,t,c,h,w]

        # transform 2D p.e. and added to features (legacy, not used)
        if self.config.model.use_neck:
            pe2d = self.neck(features)                                          # [b,t,c,h,w]
        else:
            pe2d = None

        # 2D-3D lifting
        features_3d_raw, features_3d = self.lifting(features, pe2d)             # [b,c=768,D=16,H,W], [b,c=128,D=64,H,W]

        # rendering
        results = self.render_module(features_3d, sample, return_neural_volume, render_depth, features_3d_raw)

        # return 2D features if necessary (legacy, not used)
        if self.config.model.render_feat_raw and self.training:
            results['features_2d'] = rearrange(pe2d, 'b t c h w -> (b t) c h w')
        
        return results






