import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import dataclasses
import torch.distributed as dist
import itertools
from einops import rearrange
from PIL import Image
from torchvision import transforms
import math
import itertools
from utils import exp_utils, loss_utils, vis_utils, eval_utils, geo_utils
from pytorch3d.renderer import look_at_view_transform
    
import itertools


def perform_inference(config, permute, img_path, model,
               output_dir, device, rank, mode='test'):
    model.eval()

    print('loading demo images')
    all_imgs = load_imgs(config, img_path, device)   # [n,t,c,h,w]
    print('all images shape', all_imgs.shape)

    for idx, inputs in enumerate(all_imgs):
        inputs = inputs.to(device).unsqueeze(0).float() # [1,t,c,h,w]

        if permute:
            all_permutation = generate_all_permutation(inputs.shape[1])
        else:
            all_permutation = [list(range(inputs.shape[1]))]
        
        for permu_idx, permutation in enumerate(all_permutation):
            if len(permutation)!=6:
                continue
                
            print('scene idx {}, permutation {}'.format(idx, permutation))
            cur_inputs = inputs[:, permutation]
            
            neural_volume = get_neural_volume(config, model, cur_inputs, device)
            nvs_results = get_nvs_results(config, model, neural_volume, device)
            vis_utils.save_demo_nvs_results(idx,
                                            permutation,
                                            nvs_results,
                                            output_dir=output_dir,
                                            inv_normalize=config.train.normalize_img)
    

def get_neural_volume(config, model, inputs, device):
    sample_inference = {'images': inputs}
    features, densities = model(sample_inference, device, return_neural_volume=True)
    return (features, densities)


def get_nvs_results(config, model, neural_volume, device):
    # num_views_all = 4 * 7
    # elev = torch.linspace(0, 0, num_views_all)
    # azim = torch.linspace(0, 360, num_views_all) + 180
    num_views_all = 8 * 7
    elev, azim = [], []
    elev.append(torch.linspace(0, 20, num_views_all // 8))
    azim.append(torch.linspace(0, 40, num_views_all // 8) + 180)
    elev.append(torch.linspace(20, 0, num_views_all // 8))
    azim.append(torch.linspace(40, 80, num_views_all // 8) + 180)
    elev.append(torch.linspace(0, -20, num_views_all // 8))
    azim.append(torch.linspace(80, 40, num_views_all // 8) + 180)
    elev.append(torch.linspace(-20, 0, num_views_all // 8))
    azim.append(torch.linspace(40, 0, num_views_all // 8) + 180)
    elev.append(torch.linspace(0, 20, num_views_all // 8))
    azim.append(torch.linspace(0, -40, num_views_all // 8) + 180)
    elev.append(torch.linspace(20, 0, num_views_all // 8))
    azim.append(torch.linspace(-40, -80, num_views_all // 8) + 180)
    elev.append(torch.linspace(0, -20, num_views_all // 8))
    azim.append(torch.linspace(-80, -40, num_views_all // 8) + 180)
    elev.append(torch.linspace(-20, 0, num_views_all // 8))
    azim.append(torch.linspace(-40, 0, num_views_all // 8) + 180)
    elev = torch.cat(elev)
    azim = torch.cat(azim)
    NVS_R_all, NVS_T_all = look_at_view_transform(dist=config.render.camera_z, elev=elev, azim=azim)  # [N=28,3,3], [N,3]
    NVS_pose_all = torch.cat([NVS_R_all, NVS_T_all.view(-1,3,1)], dim=-1).to(device)  # [N,3,4]

    features, densities = neural_volume     # [b,C,D,H,W]
    b = features.shape[0]
    for i in range(b):
        rendered_imgs_results, rendered_masks_results = [], []
        cur_features = features[i].unsqueeze(0).repeat(7,1,1,1,1)
        cur_densities = densities[i].unsqueeze(0).repeat(7,1,1,1,1)
        K = get_intrinsics(config)  # [3,3]
        K = K.unsqueeze(0).repeat(7,1,1).to(device)
        for pose_idx in range(8):
            cameras = {
                        'K': K,                      # [N,3,3]
                        'R': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,:3],
                        'T': NVS_pose_all[pose_idx*7: (pose_idx+1)*7,:3,3],
                    }
            render_results = model.module.render_module.render(cameras, cur_features, cur_densities)
            rendered_imgs_nvs = render_results['rgb']       # [N,3,h,w]
            rendered_masks_nvs = render_results['mask']     # [N,1,h,w]
            rendered_imgs_results.append(rendered_imgs_nvs)
            rendered_masks_results.append(rendered_masks_nvs)
        rendered_imgs_results = torch.cat(rendered_imgs_results, dim=0)
        rendered_masks_results = torch.cat(rendered_masks_results, dim=0)
    return rendered_imgs_results, rendered_masks_results
    

def get_intrinsics(config):
    focal_length = 0.5 / math.tan(0.5 * 0.6911112070083618)   # normalized with pixel
    img_size = config.dataset.img_size
    K = torch.tensor([[img_size * focal_length, 0., img_size / 2.0],
                    [0., img_size * focal_length, img_size / 2.0],
                    [0., 0., 1.]])
    return K


def generate_all_permutation(num_imgs):
    idxs = list(range(num_imgs))
    results = []
    for num in range(2, num_imgs+1):
        cur_results = list(itertools.permutations(idxs, num))
        results += cur_results
    
    sorted_results = []
    sorted_results = [[it[0]] + sorted(it[1:]) for it in results]
    sorted_results = [tuple(it) for it in sorted_results]
    sorted_results = list(set(sorted_results))
    sorted_results = [list(it) for it in sorted_results]

    permutations = list(itertools.permutations(idxs))
    return permutations#sorted_results


def load_imgs(config, path, device):
    all_files = os.listdir(path)
    scene_ids = sorted(list(set([it.split('_')[0] for it in all_files])))
    print('{} scenes found'.format(len(scene_ids)), scene_ids)
    all_imgs = []

    for idx in scene_ids:
        scene_imgs_name = [it for it in all_files if it.split('_')[0] == idx]
        scene_imgs_path = [os.path.join(path, it) for it in scene_imgs_name]
        imgs = []
        for img_path in scene_imgs_path:
            if img_path.endswith('jpg'):
                with Image.open(img_path) as img_pil:
                    img_np = np.asarray(img_pil)[:,:,:3]
            elif img_path.endswith('png'):
                with Image.open(img_path) as img_pil:
                    img_np = np.asarray(img_pil)[:,:,:3].copy()
                    mask_np = np.uint8(np.asarray(img_pil)[:,:,3:] > 0).copy()
                    img_np *= mask_np
            #img_np = pad_img(img_np, M=0.2)
            rgb = Image.fromarray(img_np[:,:,:3])
            rgb = rgb.resize((config.dataset.img_size, config.dataset.img_size), Image.Resampling.LANCZOS)
            rgb = np.asarray(rgb).transpose((2,0,1)) / 255.0   # [3,H,W]
            rgb = torch.from_numpy(rgb).to(device)
            imgs.append(rgb)
        imgs = torch.stack(imgs)    # [t,3,H,W]

        if config.train.normalize_img:
            normalization = transforms.Compose([
                #transforms.ColorJitter(brightness=0.7),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
            imgs = normalization(imgs)

        all_imgs.append(imgs)
    all_imgs = torch.stack(all_imgs)
    return all_imgs


def pad_img(img, M=0.2):
    H, W = img.shape[:2]
    pad = int(H * M)
    img_torch = torch.tensor(img).permute(2,0,1)    # [c,h,w]
    img_torch = F.pad(img_torch, [pad]*4, 'constant', 0)
    img_np = np.uint8(img_torch.permute(1,2,0).numpy())
    return img_np


def deform_imgs(imgs):
    # imgs in shape [t,c=3,h,w]
    t = imgs.shape[0]

    target_f = 0.5 / math.tan(0.5 * 0.6911112070083618) * 224
    cur_focal = 400
    cx, cy = 112, 112

    T = torch.tensor([[target_f / cur_focal, 0, 0],
                      [0, target_f / cur_focal, 0],
                      [0, 0, 1]], dtype=torch.float)
    
    H, W = imgs.shape[-2:]

    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
    homogeneous_grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).unsqueeze(0)  # shape [1, H, W, 3]

    # Duplicate the homogeneous grid for each image in the batch
    homogeneous_grid = homogeneous_grid.repeat(t, 1, 1, 1)

    # Apply the transformation and remove the last dimension
    new_homogeneous_grid = torch.matmul(homogeneous_grid, T.t()).squeeze(-1)

    # Normalize grid to [-1, 1] range
    new_homogeneous_grid[..., :2] /= new_homogeneous_grid[..., 2:]
    new_homogeneous_grid = new_homogeneous_grid.to(imgs.device).double()

    # Apply grid_sample to deform the image
    deformed_image = F.grid_sample(imgs, new_homogeneous_grid[..., :2], align_corners=False)
    return deformed_image