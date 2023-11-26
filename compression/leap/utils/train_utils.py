import os
import pprint
import random
import numpy as np
import torch
import warnings
import torch.nn as nn
import kornia
import kornia.augmentation as K
from kornia.constants import DataKey
import copy

from dataset.kubric import Kubric
from dataset.gso import GSO
from dataset.omniobject3d import Omniobject3D
from dataset.objaverse import Objaverse
from dataset.dtu import DTU


def get_optimizer(config, model):
    # get pe parameters
    pe_list = ['neck.pe_canonical', 'lifting.latent_emb']
    pe_params = []
    for it in pe_list:
        pe_params += list(filter(lambda kv: it in kv[0], model.named_parameters()))
    
    # get backbone parameters
    backbone_param = list(filter(lambda kv: 'backbone' in kv[0], model.named_parameters()))

    # other parameters
    non_other_param_names = list(map(lambda x: x[0], pe_params)) + list(map(lambda x: x[0], backbone_param))
    other_param = [(name, param) for name, param in model.named_parameters() if name not in non_other_param_names]

    # remove param names
    pe_params = list(map(lambda x: x[1], pe_params))
    backbone_param = list(map(lambda x: x[1], backbone_param))
    other_param = list(map(lambda x: x[1], other_param))

    # get optimizer
    if config.model.backbone_fix:
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW([{'params': pe_params, 'lr': config.train.lr_embeddings},
                                       {'params': other_param, 'lr': config.train.lr}],
                                      lr=config.train.lr,
                                      weight_decay=config.train.weight_decay)
    else:
        optimizer = torch.optim.AdamW([{'params': pe_params, 'lr': config.train.lr_embeddings},
                                       {'params': backbone_param, 'lr': config.train.lr_backbone},
                                       {'params': other_param, 'lr': config.train.lr}],
                                       lr=config.train.lr,
                                       weight_decay=config.train.weight_decay)
    return optimizer


def get_dataset(config, split='train'):
    name = config.dataset.name
    if name == 'kubric':
        data = Kubric(config, split=split)
    elif name == 'gso':
        data = GSO(config, split=split)
    elif name == 'omniobject3d':
        data = Omniobject3D(config, split=split)
    elif name == 'objaverse':
        data = Objaverse(config, split=split)
    elif name == 'dtu':
        data = DTU(config, split=split)
    else:
        raise NotImplementedError('not implemented dataset')
    return data


def resume_training(model, optimizer, schedular, scaler, output_dir, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    output_dir = os.path.join(output_dir, cpt_name)
    if os.path.isfile(output_dir):
        print("=> loading checkpoint {}".format(output_dir))
        if device is not None:
            checkpoint = torch.load(output_dir, map_location=device)
        else:
            checkpoint = torch.load(output_dir, map_location=torch.device('cpu'))
        
        # load model
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)

        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])

        # load schedular
        schedular.load_state_dict(checkpoint['schedular'])

        # load scaler
        scaler.load_state_dict(checkpoint['scaler'])

        # load epoch
        start_epoch = checkpoint['epoch']

        # load data
        best_psnr = checkpoint['best_psnr'] if 'best_psnr' in checkpoint.keys() else 0.0

        del checkpoint, state_dict

        return model, optimizer, schedular, scaler, start_epoch, best_psnr
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(output_dir))
    

def load_pretrain(config, model, cpt_path):
    checkpoint = torch.load(cpt_path, map_location=torch.device('cpu'))
    # load model
    if "module" in list(checkpoint["state_dict"].keys())[0]:
        state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint["state_dict"]

    state_dict_new = copy.deepcopy(state_dict)

    # # some hard coded operations to skip updampling layers
    # if config.dataset.img_size_render == config.dataset.img_size:
    #     state_dict_new['render_module.render.upsample_conv.0.weight'] = state_dict['render_module.render.upsample_conv.3.weight']
    #     state_dict_new['render_module.render.upsample_conv.0.bias'] = state_dict['render_module.render.upsample_conv.3.bias']
    #     state_dict_new['render_module.render.upsample_conv.1.weight'] = state_dict['render_module.render.upsample_conv.4.weight']
    #     state_dict_new['render_module.render.upsample_conv.1.bias'] = state_dict['render_module.render.upsample_conv.4.bias']
    #     state_dict_new['render_module.render.upsample_conv.1.running_mean'] = state_dict['render_module.render.upsample_conv.4.running_mean']
    #     state_dict_new['render_module.render.upsample_conv.1.running_var'] = state_dict['render_module.render.upsample_conv.4.running_var']
    #     state_dict_new['render_module.render.upsample_conv.3.weight'] = state_dict['render_module.render.upsample_conv.6.weight']
    #     state_dict_new['render_module.render.upsample_conv.3.bias'] = state_dict['render_module.render.upsample_conv.6.bias']
    #     del state_dict_new['render_module.render.upsample_conv.4.weight']
    #     del state_dict_new['render_module.render.upsample_conv.4.bias']
    #     del state_dict_new['render_module.render.upsample_conv.4.running_mean']
    #     del state_dict_new['render_module.render.upsample_conv.4.running_var']
    #     del state_dict_new['render_module.render.upsample_conv.4.num_batches_tracked']
    #     del state_dict_new['render_module.render.upsample_conv.6.weight']
    #     del state_dict_new['render_module.render.upsample_conv.6.bias']

    missing_states = set(model.state_dict().keys()) - set(state_dict_new.keys())
    if len(missing_states) > 0:
        warnings.warn("Missing keys ! : {}".format(missing_states))
    model.load_state_dict(state_dict_new, strict=True)

    del checkpoint, state_dict, state_dict_new
    return model


def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def init_weights_conv(m, mean_weight, mean_bias):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=mean_weight, std=1e-4)
        nn.init.normal_(m.bias, mean=mean_bias, std=1e-4)


def truncated_normal_(tensor, mean=0, std=1):
    # Generate truncated normal distribution
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()

    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def process_data(config, sample, device='cuda'):
    brightness = config.dataset.aug_brightness
    contrast = config.dataset.aug_contrast
    saturation = config.dataset.aug_saturation
    hue = config.dataset.aug_hue

    NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406])
    NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225])
    normalization = kornia.enhance.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    inverse_normalization = kornia.enhance.Denormalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    imgs = sample['images']
    assert len(imgs.shape) == 5   # [b,t,c,h,w]
    
    transform = K.AugmentationSequential(
                    K.ColorJiggle(brightness, contrast, saturation, hue, p=1.0),
                    data_keys=[DataKey.INPUT],  # Just to define the future input here.
                    same_on_batch=True,)
    
    imgs_aug = []
    for img_scene in imgs:
        img_scene_aug = inverse_normalization(img_scene)
        img_scene_aug = transform(img_scene_aug)
        img_scene_aug = normalization(img_scene_aug)
        imgs_aug.append(img_scene_aug)
    imgs_aug = torch.stack(imgs_aug)

    sample['images'] = imgs_aug
    return sample