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
from utils import exp_utils, loss_utils, vis_utils, eval_utils, geo_utils
from pytorch3d.renderer import look_at_view_transform
import lpips

logger = logging.getLogger(__name__)


def evaluation(config, loader, dataset, model,
               output_dir, device, rank, mode='test'):
    lpips_vgg = lpips.LPIPS(net="vgg").to(device)
    lpips_vgg.eval()

    model.eval()
    metrics = {'seen_psnr': [], 'seen_ssim': [], 'seen_lpips': [],
               'unseen_psnr': [], 'unseen_ssim': [], 'unseen_lpips': [],}
    time_meters = exp_utils.AverageMeters()

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            sample = exp_utils.dict_to_cuda(sample, device)
            print(sample['seq_name'])

            all_permutation = generate_permutation(config)
            all_permutation_neural_volume = {}
            all_permutation_results = []
            all_permutation_results_dict = {}
            all_permutation_nvs = {}

            for permu_idx, permutation in enumerate(all_permutation):
                cur_metrics = {'seen_psnr': [], 'seen_ssim': [], 'seen_lpips': [],
                               'unseen_psnr': [], 'unseen_ssim': [], 'unseen_lpips': [],}
                
                start = time.time()
                neural_volume = get_neural_volume(config, model, sample, permutation, device)
                all_permutation_neural_volume[str(permu_idx)] = neural_volume

                nvs_results = get_nvs_results(config, model, sample, permutation, device, neural_volume)
                all_permutation_nvs[str(permu_idx)] = nvs_results
                time_meters.add_loss_value('Inference time', time.time() - start)
                # note that the reported time using batchifyied input rather than a for loop, which is slower

                cur_metrics = eval_nvs(config, nvs_results, sample, cur_metrics, lpips_vgg, device)
                print('idx {}, permute {}'.format(batch_idx, permu_idx), cur_metrics)
                cur_metrics_save = get_metrics_save(cur_metrics)
                all_permutation_results.append((permu_idx, cur_metrics_save['psnr']))
                all_permutation_results_dict[str(permu_idx)] = cur_metrics

                # vis_utils.vis_seq(vid_clips=sample['images'][:,config.dataset.num_frame:],
                #                 vid_masks=sample['fg_probabilities'][:,config.dataset.num_frame:],
                #                 recon_clips=nvs_results[0],
                #                 recon_masks=nvs_results[1],
                #                 iter_num=str(batch_idx) + '_' + str(permu_idx),
                #                 output_dir=output_dir,
                #                 subfolder='test_seq',
                #                 inv_normalize=config.train.normalize_img)
            
            all_permutation_results_sorted = sorted(all_permutation_results, key=lambda x: x[1], reverse=True)
            best_idx = all_permutation_results_sorted[0][0]

            neural_volume = all_permutation_neural_volume[str(best_idx)]
            nvs_results = all_permutation_nvs[str(best_idx)]
            metrics_best_cur = all_permutation_results_dict[str(best_idx)]
            
            metrics = update_metrics(metrics, metrics_best_cur)
            
            generate_360_vis(config, model, neural_volume, sample, device, batch_idx, output_dir)

            # visualize input view results
            # vis_utils.vis_seq(vid_clips=sample['images'][:,:config.dataset.num_frame],
            #                   vid_masks=sample['fg_probabilities'][:,:config.dataset.num_frame],
            #                   recon_clips=input_results[0],
            #                   recon_masks=input_results[1],
            #                   iter_num=str(batch_idx),
            #                   output_dir=output_dir,
            #                   subfolder='test_seq_input',
            #                   inv_normalize=config.train.normalize_img)
            
            # visualize novel view results
            vis_utils.vis_seq(vid_clips=sample['images'][:,config.dataset.num_frame:],
                                vid_masks=sample['fg_probabilities'][:,config.dataset.num_frame:],
                                recon_clips=nvs_results[0],
                                recon_masks=nvs_results[1],
                                iter_num=str(batch_idx),
                                output_dir=output_dir,
                                subfolder='test_seq',
                                inv_normalize=config.train.normalize_img)
            
            instance_name = sample['seq_name'][0]
            vis_utils.vis_nvs_separate(imgs=nvs_results[0].squeeze(0),
                                       imgs_gt=sample['images'][0],
                                       instance_name=instance_name, 
                                       output_dir=output_dir,
                                       subfolder='test_seq_split',
                                       inv_normalize=config.train.normalize_img)

            info = 'idx {}, avg time {}: metrics: {}'.format(batch_idx, time_meters.average_meters['Inference time'].avg, metrics_best_cur)
            logger.info(info)

    metrics_mean = {k: torch.tensor(v).mean().item() for k, v in metrics.items()}
    print_results(metrics_mean)
    return metrics
    

def get_neural_volume(config, model, sample, permutation, device):
    t = config.dataset.num_frame        # number of input
    imgs = sample['images'][:,:t][:,permutation]
    masks = sample['fg_probabilities'][:,:t][:,permutation]

    sample_inference = {'images': imgs[:,:t]}
    features, densities = model(sample_inference, device, return_neural_volume=True)
    return (features, densities)


def get_nvs_results(config, model, sample, permutation, device, neural_volume):
    t = config.dataset.num_frame        # number of input
    t_all = sample['images'].shape[1]
    t_nvs = t_all - t

    features, densities = neural_volume     # [b,C,D,H,W]
    b,C,D,H,W = features.shape

    # parse NVS cameras
    camK = sample['K_cv2'][:,t:].to(device)                                  # [b,t_nvs,3,3]
    #camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'][:,t:].to(device)   # [b,t_nvs,4,4]
    camE_cv2 = permutation_transform_camera_pose(config, sample, device, permutation)
    camera_params = {
        'R': camE_cv2.reshape(-1,4,4)[:,:3,:3],    # [b*t_nvs,3,3]
        'T': camE_cv2.reshape(-1,4,4)[:,:3,3],     # [b*t,3]
        'K': camK.reshape(-1,3,3)                  # [b*t,3,3]
    }

    # repeat neural volume for all frame in t
    densities_all = densities.unsqueeze(1).repeat(1,t_nvs,1,1,1,1).reshape(b*t_nvs,1,D,H,W)
    features_all = features.unsqueeze(1).repeat(1,t_nvs,1,1,1,1).reshape(b*t_nvs,C,D,H,W)

    # render novel views
    render_results = model.module.render_module.render(camera_params, features_all, densities_all)
    rendered_imgs_nvs = render_results['rgb']       # [b*t_nvs,3,h,w]
    rendered_masks_nvs = render_results['mask']     # [b*t_nvs,1,h,w]

    rendered_imgs_nvs = rearrange(rendered_imgs_nvs, '(b t) c h w -> b t c h w', b=b, t=t_nvs)
    rendered_masks_nvs = rearrange(rendered_masks_nvs, '(b t) c h w -> b t c h w', b=b, t=t_nvs)
    return (rendered_imgs_nvs, rendered_masks_nvs)


def eval_nvs(config, nvs_results, sample, metrics, lpips_vgg, device):
    nvs_imgs, nvs_masks = nvs_results       # [b,t_nvs,c,h,w]
    t_nvs = nvs_imgs.shape[1]
    t = config.dataset.num_frame            # number of input
    target_imgs = sample['images'][:,t:]    # [b,t_nvs,c,h,w]
    seen_flag = sample['seen_flag']         # [b]
    b = target_imgs.shape[0]

    if config.train.normalize_img:
        nvs_imgs = vis_utils.unnormalize(nvs_imgs).clip(min=0.0, max=1.0)
        nvs_masks = nvs_masks.clip(min=0.0, max=1.0)
        target_imgs = vis_utils.unnormalize(target_imgs)

    for i in range(b):
        cur_nvs_imgs = nvs_imgs[i]          # [t_nvs,c,h,w]
        cur_target_imgs = target_imgs[i]
        cur_seen_flag = seen_flag[i].item()
        cur_psnr, cur_ssim = [], []
        for j in range(t_nvs):
            nvs = cur_nvs_imgs[j].permute(1,2,0).detach().cpu().numpy()
            tgt = cur_target_imgs[j].permute(1,2,0).detach().cpu().numpy()
            psnr, ssim = eval_utils.compute_img_metric(nvs, tgt)
            cur_psnr.append(psnr)
            cur_ssim.append(ssim)
        cur_lpips = lpips_vgg(cur_nvs_imgs, cur_target_imgs).mean().item()
        cur_psnr = np.mean(cur_psnr)
        cur_ssim = np.mean(cur_ssim)
        if cur_seen_flag > 0:
            metrics['seen_psnr'].append(cur_psnr)
            metrics['seen_ssim'].append(cur_ssim)
            metrics['seen_lpips'].append(cur_lpips)
        else:
            metrics['unseen_psnr'].append(cur_psnr)
            metrics['unseen_ssim'].append(cur_ssim)
            metrics['unseen_lpips'].append(cur_lpips)

    return metrics


def generate_360_vis(config, model, neural_volume, sample, device, batch_idx, output_dir):
    num_views_all = 4 * 7
    if config.train.use_uncanonicalized_pose:
        elev = torch.linspace(0, 0, num_views_all) + 45
        azim = torch.linspace(0, 360, num_views_all)
    else:
        elev = torch.linspace(0, 0, num_views_all)
        azim = torch.linspace(0, 360, num_views_all) + 180
    if config.dataset.name == 'dtu':
        elev, azim = [], []
        elev.append(torch.linspace(0, 10, num_views_all // 4))
        azim.append(torch.linspace(0, 15, num_views_all // 4) + 180)
        elev.append(torch.linspace(10, 0, num_views_all // 4))
        azim.append(torch.linspace(15, 30, num_views_all // 4) + 180)
        elev.append(torch.linspace(0, -10, num_views_all // 4))
        azim.append(torch.linspace(30, 15, num_views_all // 4) + 180)
        elev.append(torch.linspace(-10, 0, num_views_all // 4))
        azim.append(torch.linspace(15, 0, num_views_all // 4) + 180)
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
        for pose_idx in range(4):
            cameras = {
                        'K': sample['K_cv2'][i][0:1].repeat(7,1,1),                      # [N,3,3]
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
        vis_utils.vis_NVS(imgs=rendered_imgs_results,
                          masks=rendered_masks_results, 
                          img_name=str(batch_idx) + '_' + str(i),
                          output_dir=output_dir,
                          subfolder='test_seq',
                          inv_normalize=config.train.normalize_img)
    

def generate_permutation(config):
    t = config.dataset.num_frame
    all_permutation = []
    for i in range(t):
        if i == 0:
            permute = list(range(t))
        elif i == (t-1):
            permute = [i] + list(range(t-1))
        else:
            permute = [i] + list(range(i)) + list(range(i+1, t))
        all_permutation.append(permute)
    return all_permutation


def get_metrics_save(metrics):
    if len(metrics['seen_psnr']) != 0:
        return {'psnr': metrics['seen_psnr'][0], 'ssim': metrics['seen_ssim'][0], 'lpips': metrics['seen_lpips'][0]}
    else:
        return {'psnr': metrics['unseen_psnr'][0], 'ssim': metrics['unseen_ssim'][0], 'lpips': metrics['unseen_lpips'][0]}
    

def update_metrics(metrics, cur_metrics):
    for k in metrics.keys():
        metrics[k] = metrics[k] + cur_metrics[k]
    return metrics


def print_results(metrics):
    print('------------------------Evaluation results------------------------')
    for k in metrics.keys():
        print('     {}: {}'.format(k, metrics[k]))


def permutation_transform_camera_pose(config, sample, device, permutation):
    '''
    permutation: list of permutation index
    '''
    t = config.dataset.num_frame        # number of input
    t_all = sample['images'].shape[1]
    t_nvs = t_all - t

    if not config.train.use_uncanonicalized_pose:
        tmp = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, -config.render.camera_z],
                            [0.0, 0.0, 0.0, 1.0]]).to(device)
        
        camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'].to(device).squeeze()
        camE_cv2[:t] = camE_cv2[:t][permutation]
        camPose_cv2 = torch.inverse(camE_cv2)

        camPose_cv2_rel = geo_utils.get_relative_pose(camPose_cv2[0], camPose_cv2)   # [t_all,4,4]
        camPose_cv2_transform = geo_utils.canonicalize_poses(tmp, camPose_cv2_rel.to(device))
        camE_cv2_transform = torch.inverse(camPose_cv2_transform).unsqueeze(0)

        #__import__('pdb').set_trace()
        return camE_cv2_transform[:,t:]
    else:
        camE_cv2 = sample['cam_extrinsics_cv2_canonicalized'].to(device)[:,t:]
        return camE_cv2

    