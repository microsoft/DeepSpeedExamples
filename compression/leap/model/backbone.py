import torch.nn as nn
import torch
import torch.nn.functional as F
from .third_party_transformer.modeling_dinov2 import Dinov2Backbone
from .third_party_transformer.modeling_dinov2_config import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Backbone as Dinov2BackboneHF
def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dinov2':
        assert type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        #backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(type))
        #config = 
        #config  = 'facebook/dinov2-base'
        with torch.no_grad():
            pretrainedViT_parameter = Dinov2BackboneHF.from_pretrained('facebook/dinov2-base').state_dict()
            
        config = Dinov2Config(hidden_size=config.model.backbone_out_dim, \
                            module_camera_feature=config.model.module_camera_feature,\
                            image_size=config.dataset.img_size, patch_size=config.model.backbone_path_size)
        backbone = Dinov2Backbone(config)
        backbone.load_state_dict(pretrainedViT_parameter, strict=False)
        
        del pretrainedViT_parameter
        down_rate = 14
        if type == 'vitb14':
            backbone_dim = 768
        elif type == 'vits14':
            backbone_dim = 384
    ##### backbone.hidden_states
    return backbone, down_rate, backbone_dim



class BackboneOutBlock(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(BackboneOutBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if out_dim != in_dim:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_dim, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(512, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(inplace=True),
                )
    
    def forward(self, x):
        if self.in_dim != self.out_dim:
            x = self.out_conv(x)
        return x

