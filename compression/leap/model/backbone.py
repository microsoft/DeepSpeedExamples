import torch.nn as nn
import torch
import torch.nn.functional as F


def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dinov2':
        assert type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(type))
        down_rate = 14
        if type == 'vitb14':
            backbone_dim = 768
        elif type == 'vits14':
            backbone_dim = 384
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

