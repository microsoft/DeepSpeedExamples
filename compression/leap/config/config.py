import yaml
import os
import numpy as np
from easydict import EasyDict as edict

config = edict()

# experiment config
config.exp_name = 'leap'
config.exp_group = 'none'
config.output_dir = './output/'
config.log_dir = './log'
config.workers = 8
config.print_freq = 100
config.vis_freq = 300
config.eval_vis_freq = 20
config.seed = 0

# dataset config
config.dataset = edict()
config.dataset.name = 'kubric'
config.dataset.category = 'general'
config.dataset.task = 'multisequence'
config.dataset.img_size = 512
config.dataset.img_size_height = 0
config.dataset.img_size_render = 256
config.dataset.num_frame = 5
config.dataset.frame_interval = 5
config.dataset.mask_images = False
config.dataset.train_all_frame = False
config.dataset.train_shuffle = False
config.dataset.augmentation = False
config.dataset.aug_brightness = 0
config.dataset.aug_contrast = 0
config.dataset.aug_saturation = 0
config.dataset.aug_hue = 0

# network config
config.model = edict()
config.model.norm_first = False
# backbone
config.model.backbone_name = 'dinov2'
config.model.backbone_type = 'vitb14'
config.model.backbone_fix = False
config.model.backbone_out_dim = 256
# encoder
config.model.encoder_layers = 3
# pe transformer
config.model.use_neck = True
config.model.neck_scale = 'constant_1'
config.model.neck_layers = 3
config.model.pe_with_spatial_pe = False
# lifting
config.model.lifting_TXdecoder_permute = False
config.model.use_pe_lifting = False
config.model.lifting_use_conv3d = False
config.model.lifting_layers = 6
config.model.latent_res = 32
# rendering
config.model.volume_res = 64
config.model.render_feat_dim = 16
config.model.render_feat_raw = False
# others
config.model.rot_representation = 'euler'
config.model.use_flash_attn = False

# render config
config.render = edict()
config.render.n_pts_per_ray = 200
config.render.volume_size = 1.0  #  in meters
config.render.min_depth = 0.1
config.render.max_depth = 1.2
config.render.camera_z = 0.6  # camera pose T_z
config.render.camera_focal = 250
config.render.k_size = 5

# loss config
config.loss = edict()
config.loss.weight_render_rgb = 1.0
config.loss.weight_render_mask = 0.2
config.loss.weight_perceptual = 0.0
config.loss.iter_perceptual = 0
config.loss.weight_feat_render = 0.0

# training config
config.train = edict()
config.train.resume = False
config.train.lr = 0.0001
config.train.lr_embeddings = 0.0001
config.train.lr_backbone = 0.00005
config.train.weight_decay = 0.0001
config.train.schedular_warmup_iter = 100
config.train.total_iteration = 200000
config.train.batch_size = 16
config.train.accumulation_step = 2
config.train.normalize_img = True
config.train.grad_max = 1.0
config.train.use_amp = False
config.train.use_rand_view = False
config.train.min_rand_view = 3
config.train.use_uncanonicalized_pose = False
config.train.pretrain_path = ''


# test config
config.test = edict()
config.test.batch_size = 1
config.test.compute_metric = True


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                     config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)