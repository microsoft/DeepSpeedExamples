import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import imageio
import os
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import torchvision.transforms as transforms


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=1/std),
    transforms.Normalize(mean=-mean, std=[1, 1, 1])
])

transform = transforms.Compose([
    transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
    ])


def vis_seq(vid_clips, vid_masks, recon_clips, recon_masks, iter_num, output_dir, 
            subfolder='train_seq', vid_depths=None, recon_depths=None, inv_normalize=False):
    '''
    vid_clips: in shape [b,t,c,h,w]
    '''
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    vid_clips = vid_clips.float().detach().cpu()        # [B,t,c,h,w]
    recon_clips = recon_clips.float().detach().cpu()
    vid_masks = vid_masks.float().detach().cpu()
    recon_masks = recon_masks.float().detach().cpu()

    if inv_normalize:
        b,t,c,h,w = vid_clips.shape
        vid_clips = rearrange(vid_clips, 'b t c h w -> (b t) c h w')
        recon_clips = rearrange(recon_clips, 'b t c h w -> (b t) c h w')
        vid_clips = inverse_transform(vid_clips)
        recon_clips = inverse_transform(recon_clips)
        vid_clips = rearrange(vid_clips, '(b t) c h w -> b t c h w', b=b, t=t)
        recon_clips = rearrange(recon_clips, '(b t) c h w -> b t c h w', b=b, t=t)
    
    vid_clips = vid_clips.clip(min=0.0, max=1.0)
    recon_clips = recon_clips.clip(min=0.0, max=1.0)

    addition_col, depth_diff_col = 0, 0
    if torch.is_tensor(recon_depths):
        recon_depths = recon_depths.detach().cpu()
        addition_col += 1
    if torch.is_tensor(vid_depths):
        vid_depths = vid_depths.detach().cpu()
        addition_col += 1
    if torch.is_tensor(vid_depths) and torch.is_tensor(recon_depths):
        depth_diff_col = 1

    #__import__('pdb').set_trace()

    B = vid_clips.shape[0]
    for i in range(B):
        save_name = os.path.join(output_dir, str(iter_num) + '_' + str(i) + '.jpg')
        rows = vid_clips.shape[1]
        fig = plt.figure(figsize=(12, 12))
        fig.clf()
        col_nb = 4 + addition_col + depth_diff_col # Col 1: img, 2: rendered img, 3: mask, 4: rendered mask, 5 (5-7): depths and rendered depth

        for j in range(rows):
            img = vid_clips[i][j].permute(1,2,0).numpy()            # [h,w,c]
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 1)
            ax.imshow(img)
            ax.axis("off")

            rendered_img = recon_clips[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 2)
            ax.imshow(rendered_img)
            ax.axis('off')

            mask = vid_masks[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 3)
            ax.imshow(mask)
            ax.axis("off")

            rendered_mask = recon_masks[i][j].permute(1,2,0).numpy()
            ax = fig.add_subplot(rows, col_nb, j * col_nb + 4)
            ax.imshow(rendered_mask)
            ax.axis('off')

            if torch.is_tensor(vid_depths):
                depth = vid_depths[i][j].permute(1,2,0).numpy()
                ax = fig.add_subplot(rows, col_nb, j * col_nb + col_nb - 1 - depth_diff_col)
                ax.imshow(depth, cmap='magma', vmin=0, vmax=2)
                ax.axis('off')
            
            if torch.is_tensor(recon_depths):
                rendered_depth = recon_depths[i][j].permute(1,2,0).numpy()
                ax = fig.add_subplot(rows, col_nb, j * col_nb + col_nb - depth_diff_col)
                ax.imshow(rendered_depth, cmap='magma', vmin=0, vmax=2)
                ax.axis('off')

            if torch.is_tensor(recon_depths) and torch.is_tensor(vid_depths):
                ax = fig.add_subplot(rows, col_nb, j * col_nb + col_nb)
                ax.imshow(np.abs(depth - rendered_depth), cmap='magma', vmin=0, vmax=2)
                ax.axis('off')
        
        plt.savefig(save_name, dpi=100)
        plt.close('all')


def vis_NVS(imgs, masks, img_name, output_dir, inv_normalize=False, subfolder='val_seq', depths=None):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)
    save_name = os.path.join(output_dir, str(img_name) + '.gif')

    imgs = imgs.float().detach().cpu()  # [N,c,h,w]
    if inv_normalize:
        imgs = inverse_transform(imgs)
    imgs = imgs.clip(min=0.0, max=1.0)
    masks = masks.clip(min=0.0, max=1.0)

    masks = masks.float().detach().cpu()  # [N,1,h,w]
    masks = masks.repeat(1,3,1,1)
    if torch.is_tensor(depths):
        depths = depths.detach().cpu()  # [N,1,h,w]
        depths = depths.repeat(1,3,1,1)
        imgs = 255 * torch.cat([imgs, masks, depths], dim=-1)  # [N,c,h, 3*w]
    else:
        imgs = 255 * torch.cat([imgs, masks], dim=-1)  # [N,c,h, 2*w]
    imgs = imgs.clip(min=0.0, max=255.0)
        
    frames = [np.uint8(img.permute(1,2,0).numpy()) for img in imgs]  # image in [h,w,c]
    #from IPython import embed; embed()
    imageio.mimsave(save_name, frames, 'GIF', duration=0.1)


def vis_nvs_separate(imgs, imgs_gt, instance_name, output_dir, subfolder='test_seq_split', inv_normalize=False):
    '''
    imgs: in shape [n,c,h,w] with value range [0,1]
    '''
    #breakpoint()
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    if inv_normalize:
        imgs = inverse_transform(imgs)
        imgs_gt = inverse_transform(imgs_gt)

    num_nvs = imgs.shape[0]

    imgs_255 = (imgs * 255).clamp(min=0.0, max=255.0).permute(0,2,3,1).detach().cpu().numpy()   # [n,h,w,c]
    imgs_gt = (imgs_gt * 255).clamp(min=0.0, max=255.0).permute(0,2,3,1).detach().cpu().numpy()

    for i, img_255 in enumerate(imgs_gt):
        img_255_uint8 = np.uint8(img_255)
        os.makedirs(os.path.join(output_dir, instance_name), exist_ok=True)
        save_path = os.path.join(output_dir, instance_name, '{}_gt.png'.format(i))
        cv2.imwrite(save_path, img_255_uint8[:,:,::-1])

    for i, img_255 in enumerate(imgs_255):
        img_255_uint8 = np.uint8(img_255)
        os.makedirs(os.path.join(output_dir, instance_name), exist_ok=True)
        save_path = os.path.join(output_dir, instance_name, '{}.png'.format(i+5))
        cv2.imwrite(save_path, img_255_uint8[:,:,::-1])


def save_demo_nvs_results(scene_idx, permutation, nvs_results, output_dir, inv_normalize=False, white_bg=False):
    imgs, masks = nvs_results
    #breakpoint()
    output_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    img_name = 'scene_{}_permutation_'.format(scene_idx) + '_'.join(map(str, permutation)) + '.gif'
    save_name = os.path.join(output_dir, img_name)

    imgs = imgs.float().detach().cpu()  # [N,c,h,w]
    if inv_normalize:
        imgs = inverse_transform(imgs)
    imgs = imgs.clip(min=0.0, max=1.0)
    masks = masks.clip(min=0.0, max=1.0).float().detach().cpu()

    if white_bg:
        imgs = imgs + (1. - masks)

    masks = masks.float().detach().cpu()  # [N,1,h,w]
    masks = masks.repeat(1,3,1,1)
    imgs = 255 * torch.cat([imgs, masks], dim=-1)  # [N,c,h, 2*w]
    imgs = imgs.clip(min=0.0, max=255.0)
        
    frames = [np.uint8(img.permute(1,2,0).numpy()) for img in imgs]  # image in [h,w,c]
    #frames = [cv2.resize(frame, (1024, 512)) for frame in frames]
    #from IPython import embed; embed()
    imageio.mimsave(save_name, frames, 'GIF', duration=0.1)


def unnormalize(img):
    '''
    img in [b,t,c,h,w] or [b,c,h,w]
    '''
    if len(img.shape) == 5:
        b,t,c,h,w = img.shape
        img = rearrange(img, 'b t c h w -> (b t) c h w')
        img = inverse_transform(img)
        img = rearrange(img, '(b t) c h w -> b t c h w', b=b, t=t)
    else:
        img = inverse_transform(img)
    return img


def normalize(img):
    '''
    img in [b,t,c,h,w] or [b,c,h,w]
    '''
    if len(img.shape) == 5:
        b,t,c,h,w = img.shape
        img = rearrange(img, 'b t c h w -> (b t) c h w')
        img = transform(img)
        img = rearrange(img, '(b t) c h w -> b t c h w', b=b, t=t)
    else:
        img = transform(img)
    return img