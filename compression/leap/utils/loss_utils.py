import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_losses(config, iter_num, pred, sample, perceptual_loss=None):
    losses = {}
    rendered_imgs, rendered_masks = pred['rgb'], pred['mask']   # [b*t,c,h,w]
    device = rendered_imgs.device

    if not config.dataset.train_all_frame:
        imgs = sample['images'].to(device)                          # [b,t,c,h,w]
        masks = sample['fg_probabilities'].to(device)
    else:
        t_input = config.dataset.num_frame
        imgs = sample['images'][:,t_input:].to(device)
        masks = sample['fg_probabilities'][:,t_input:].to(device)
    b,t,c,h,w = imgs.shape
    target_imgs = imgs.reshape(b*t,c,h,w)
    target_masks = masks.reshape(b*t,1,h,w)
    #__import__('pdb').set_trace()

    loss_rgb = F.mse_loss(rendered_imgs, target_imgs)
    loss_mask = F.mse_loss(rendered_masks, target_masks)
    if perceptual_loss is not None and iter_num > config.loss.iter_perceptual:
        loss_perceptual = perceptual_loss(rendered_imgs, target_imgs, already_normalized=config.train.normalize_img)
        losses['loss_perceptual'] = loss_perceptual
    else:
        loss_perceptual = 0.0

    losses.update({
        'loss_render_rgb': loss_rgb,
        'loss_render_mask': loss_mask,
        'weight_render_rgb': config.loss.weight_render_rgb,
        'weight_render_mask': config.loss.weight_render_mask,
        'weight_perceptual': config.loss.weight_perceptual
    })

    if config.model.render_feat_raw:
        loss_feat_render = compute_feat_loss(sample, pred, device)
        losses.update({
            'loss_feat_render': loss_feat_render,
            'weight_feat_render': config.loss.weight_feat_render
        })

    return losses


def compute_feat_loss(sample, pred, device):
    '''
    pe_pred and pe_render: in shape [b*t,c,h,w]
    '''
    b,t,c,h,w = sample['images'].shape
    features_2d = pred['features_2d'].detach()                          # [b*t,c',h',w']
    features_2d_render = pred['features_2d_render']

    h2, w2 = features_2d.shape[-2:]
    masks = sample['fg_probabilities'].to(device).reshape(b*t,1,h,w)    # [b,t,h,w]
    masks_down = F.interpolate(masks, [h2, w2], mode='nearest')         # [b*t,1,h',w']

    loss_feat_render = F.mse_loss(features_2d_render * masks_down.float(),
                                  features_2d * masks_down.float())

    return loss_feat_render

