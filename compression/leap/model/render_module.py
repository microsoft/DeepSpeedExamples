import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
import math
from model.volume_render import VolRender
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection


class RenderModule(nn.Module):
    def __init__(self, config, feat_res=16) -> None:
        super(RenderModule, self).__init__()

        self.config = config
        self.num_img_train = config.dataset.num_frame   # number of input images

        # feature and density layers
        self.density_head = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            #nn.LayerNorm([32,64,64,64]),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 8, 3, padding=1),
            #nn.LayerNorm([8,64,64,64]),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.render_feat_dim = config.model.render_feat_dim
        self.features_head = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            #nn.LayerNorm([32,64,64,64]),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, self.render_feat_dim, 3, padding=1),
            #nn.LayerNorm([self.render_feat_dim,64,64,64]),
            nn.BatchNorm3d(self.render_feat_dim),
            nn.LeakyReLU(inplace=True),
        )

        # build volume renderer
        self.render = VolRender(config, feat_res=feat_res)


    def forward(self, features, sample, return_neural_volume=False, render_depth=False, feat3d_raw=None):
        '''
        feat3d: in shape [b,C,D,H,W]
        '''
        b,C,D,H,W = features.shape
        device = features.device

        # get neural volume for NeRF rendering
        densities = self.density_head(features).clip(min=0.0, max=1.0-1e-5)      # [b,1,D2,H2,W2]
        features = self.features_head(features)                                  # [b,C2=16,D2,H2,W2]
        _,C2,D2,H2,W2 = features.shape
        
        if return_neural_volume:
            return (features, densities)

        # get camera pose and intrinsics for rendering
        if not self.config.dataset.train_all_frame:
            t = self.num_img_train
            camK = sample['K_cv2'][:,:t].to(device)                                  # [b,t,3,3]
            camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'][:,:t].to(device)   # [b,t,4,4]
            camera_params = {
                'R': camE_cv2.reshape(b*t,4,4)[:,:3,:3],    # [b*t,3,3]
                'T': camE_cv2.reshape(b*t,4,4)[:,:3,3],     # [b*t,3]
                'K': camK.reshape(b*t,3,3)                  # [b*t,3,3]
            }
            t_repeat = t
        else:
            # use novel views for rendering loss
            t = self.num_img_train
            camK = sample['K_cv2'][:,t:].to(device)                                  # [b,t,3,3]
            camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'][:,t:].to(device)   # [b,t,4,4]
            t_nvs = sample['K_cv2'].shape[1] - t
            camera_params = {
                'R': camE_cv2.reshape(b*t_nvs,4,4)[:,:3,:3],    # [b*t_nvs,3,3]
                'T': camE_cv2.reshape(b*t_nvs,4,4)[:,:3,3],     # [b*t_nvs,3]
                'K': camK.reshape(b*t_nvs,3,3)                  # [b*t_nvs,3,3]
            }
            t_repeat = t_nvs

        # repeat neural volume for all frame in t
        densities_all = densities.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t_repeat,1,D2,H2,W2)
        features_all = features.unsqueeze(1).repeat(1,t,1,1,1,1).reshape(b*t_repeat,C2,D2,H2,W2)

        render_results = self.render(camera_params, features_all, densities_all, render_depth, feat3d_raw)

        return render_results




        



