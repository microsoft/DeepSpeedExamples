import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Volumes
from pytorch3d.renderer import VolumeRenderer, NDCGridRaysampler, EmissionAbsorptionRaymarcher
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.renderer.cameras import PerspectiveCameras
from torch.cuda.amp import autocast
from utils.train_utils import init_weights_conv
import math
from utils.vis_utils import unnormalize, normalize
from einops import rearrange
import copy


class VolRender(nn.Module):
    def __init__(self, config, feat_res=16):
        super(VolRender, self).__init__()
        self.config = config

        # render image resolution setting
        self.img_input_res = config.dataset.img_size
        self.img_render_res = config.dataset.img_size_render
        self.render_down_rate = self.img_input_res // self.img_render_res
        self.feat_res = feat_res    # used for render p.e. (image features)
        self.render_feat_down_rate = self.img_render_res // self.feat_res
        if config.dataset.name == 'dtu':
            self.img_input_res_height = config.dataset.img_size_height
            self.img_render_res_height = self.img_input_res_height // self.render_down_rate
        else:
            self.img_input_res_height = self.img_input_res
            self.img_render_res_height = self.img_render_res
        #__import__('pdb').set_trace()

        # neural volume physical world settings
        self.volume_physical_size = config.render.volume_size

        # build image renderer
        self.raySampler = NDCGridRaysampler(image_width=self.img_render_res,
                                            image_height=self.img_render_res_height,
                                            n_pts_per_ray=config.render.n_pts_per_ray,
                                            min_depth=config.render.min_depth,
                                            max_depth=config.render.max_depth)
        self.rayMarcher = EmissionAbsorptionRaymarcher()
        self.renderer = VolumeRenderer(raysampler=self.raySampler, raymarcher=self.rayMarcher)

        # build feature2rgb upsample module
        self.upsample_conv = render_make_upconv_layers(config, self.render_down_rate)

        # build p.e. renderer
        if config.model.render_feat_raw:
            self.raySampler_feat = NDCGridRaysampler(image_width=self.feat_res,
                                                    image_height=self.feat_res,
                                                    n_pts_per_ray=config.model.latent_res,
                                                    min_depth=config.render.min_depth,
                                                    max_depth=config.render.max_depth)
            self.rayMarcher_feat = EmissionAbsorptionRaymarcher()
            self.renderer_feat = VolumeRenderer(raysampler=self.raySampler_feat, raymarcher=self.rayMarcher_feat)


    def forward(self, camera_params_in, features, densities, render_depth=False, feat3d_raw=None):
        '''
        camera_params: pytorch3d perspective camera, parameters in batch size B
        features: [B,C,D,H,W]
        densities: [B,1,D,H,W]
        pe_volume: [B,C2,D2,H2,W2]
        '''
        B,C,D,H,W = features.shape
        device = features.device
        camera_params = copy.deepcopy(camera_params_in)

        # parse camera parameters considering render downsample rate
        #__import__('pdb').set_trace()
        camera_params['K'] /= self.render_down_rate
        camera_params['K'][:,-1,-1] = 1.0
        cameras = cameras_from_opencv_projection(R=camera_params['R'],
                                                 tvec=camera_params['T'], 
                                                 camera_matrix=camera_params['K'],
                                                 image_size=torch.tensor([self.img_render_res_height, self.img_render_res]).unsqueeze(0).repeat(B,1)).to(device)
        
        # parse neural volume physical world settings
        if self.config.dataset.name == 'dtu':
            single_voxel_size = self.volume_physical_size / D
            single_voxel_size = [single_voxel_size, single_voxel_size, single_voxel_size]  # [width, height, depth]
            translation = [0,0,0]   # [+left, +up, +far]
            volume = Volumes(densities=densities,
                            features=features,
                            voxel_size=single_voxel_size,
                            volume_translation=translation)
        else:
            single_voxel_size = self.volume_physical_size / D
            volume = Volumes(densities=densities,
                            features=features,
                            voxel_size=single_voxel_size)
        
        # perform neural rendering
        rendered = self.renderer(cameras=cameras, volumes=volume, render_depth=render_depth)[0]  # [B,H,W,C+1]

        # split into rgb, mask and depth, and get to original input resolution
        if not render_depth:
            rendered_imgs, rendered_mask = rendered.split([C,1], dim=-1)
        else:
            rendered_imgs, rendered_mask, rendered_depth = rendered.split([C,1,1], dim=-1)
            rendered_depth = rendered_depth.permute(0,3,1,2).contiguous()
            rendered_depth = F.upsample(rendered_depth, size=[self.img_input_res]*2, mode='bilinear')
        #__import__('pdb').set_trace()
        rendered_imgs = rendered_imgs.permute(0,3,1,2).contiguous().float()
        rendered_mask = rendered_mask.permute(0,3,1,2).contiguous()
        rendered_imgs = self.upsample_conv(rendered_imgs)
        if self.config.train.normalize_img:
            rendered_imgs = unnormalize(rendered_imgs)
        rendered_imgs = F.relu(rendered_imgs)
        if self.config.train.normalize_img:
            rendered_imgs = normalize(rendered_imgs)
        #__import__('pdb').set_trace()
        rendered_mask = F.upsample(rendered_mask, size=[self.img_input_res_height, self.img_input_res], mode='bilinear')

        results = {
            'rgb': rendered_imgs,       # [B=b*t,3,h,w]
            'mask': rendered_mask,      # [B,1,h,w] 
        }
        if render_depth:
            results['depth']: render_depth  # [B,1,h,w]

        # render raw 3D volume to 2D
        if self.config.model.render_feat_raw and self.training and torch.is_tensor(feat3d_raw):
            camera_params['K'] /= self.render_feat_down_rate
            camera_params['K'][:,-1,-1] = 1.0
            cameras_feat = cameras_from_opencv_projection(R=camera_params['R'],
                                                          tvec=camera_params['T'], 
                                                          camera_matrix=camera_params['K'],
                                                          image_size=torch.tensor([self.feat_res]*2).unsqueeze(0).repeat(B,1)).to(device)
            single_voxel_size_feat = self.volume_physical_size / self.config.model.latent_res
            densities_feat = F.interpolate(densities, [self.config.model.latent_res]*3, mode='trilinear')
            #__import__('pdb').set_trace()
            feat3d_raw = feat3d_raw.unsqueeze(1).repeat(1,self.config.dataset.num_frame,1,1,1,1)
            feat3d_raw = rearrange(feat3d_raw, 'b t c d h w -> (b t) c d h w')
            volume_feat = Volumes(densities=densities_feat,
                                  features=feat3d_raw,
                                  voxel_size=single_voxel_size_feat)
            rendered = self.renderer_feat(cameras=cameras_feat, volumes=volume_feat, render_depth=False)[0]
            rendered_feat, _ = rendered.split([self.config.model.backbone_out_dim,1], dim=-1)
            rendered_feat = rendered_feat.permute(0,3,1,2).contiguous().float()
            results['features_2d_render'] = rendered_feat   # [B,c',h',w']

        return results


def render_make_upconv_layers(config, render_down_rate):
    k_size = config.render.k_size
    pad_size = k_size // 2
    render_feat_dim = config.model.render_feat_dim
    upsample_conv = []
    for _ in range(int(math.log2(render_down_rate))):
        upsample_conv.extend([
                nn.ConvTranspose2d(render_feat_dim, render_feat_dim, kernel_size=k_size+1, stride=2, padding=pad_size),
                #nn.LayerNorm([self.render_feat_dim,224,224]), 
                nn.BatchNorm2d(render_feat_dim),
                nn.LeakyReLU(inplace=True),])
    upsample_conv.extend([
            nn.Conv2d(render_feat_dim, 8, kernel_size=k_size, stride=1, padding=pad_size),
            #nn.LayerNorm([8,224,224]),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=k_size, stride=1, padding=pad_size),])
    upsample_conv = nn.Sequential(*nn.ModuleList(upsample_conv))
    return upsample_conv
