import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import argparse
import wandb
import transformers

from config.config import config, update_config
from model.model import LEAP
from model.external.perceptual_loss import VGGPerceptualLoss
from utils import dist_utils, exp_utils, train_utils
from scripts.trainer import train_epoch
from scripts.eval import evaluation



def parse_args():
    parser = argparse.ArgumentParser(description='Test LEAP')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument(
        '--cpt', 
        #default='output/kubric/train_224/base_res224_lr2x_percploss-0.1_res32_encoder-nonfix-1e-5_no-neck_no-feat-render-loss/cpt_last.pth.tar',
        default='output/omniobject3d/train_224/base_res224_lr2x_percploss-0.1_res32_encoder-nonfix-1e-5_layer-2-0-4_volume-2_rel-pose/cpt_best_psnr_29.21728228794307.pth.tar', 
        #default='output/dtu/train_224/base_res224_lr2x_res32_encoder-1e-5_layer-2-0-4_volume-2.0_iter3k_pretrain-omniobj/cpt_best_psnr_17.932848452595916.pth.tar',
        #default='output/objaverse/train_224/base_res224_lr1x_percploss-0.2_res32_encoder-nonfix-1e-5_layer-2-0-4_volume-1.4_20k-iter_pretrain-kubric/cpt_best_psnr_25.72105374709138.pth.tar',
        type=str, help='checkpoint path')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='eval')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    logger.info('checkpoint path: {}'.format(args.cpt))
    output_dir = os.path.join(output_dir, config.dataset.name, config.dataset.category)

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # set device
    gpus = range(torch.cuda.device_count())
    distributed = torch.cuda.device_count() > 1
    device = torch.device('cuda') if len(gpus) > 0 else torch.device('cpu')
    if "LOCAL_RANK" in os.environ:
        dist_utils.dist_init(int(os.environ["LOCAL_RANK"]))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # get model
    model = LEAP(config).to(device)
    
    # resume training
    state_dict = torch.load(args.cpt)['state_dict']
    model.load_state_dict(state_dict, strict=True)

    # distributed training
    ddp = False
    model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        find_unused = True if (not config.model.backbone_fix) else False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=find_unused)
        device_num = len(device_ids)
        ddp = True

    # get dataset and dataloader
    val_data = train_utils.get_dataset(config, split='test')
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=config.test.batch_size, 
                                             shuffle=False,
                                             num_workers=int(config.workers), 
                                             pin_memory=True, 
                                             drop_last=False)
    

    eval_results = evaluation(config,
                            loader=val_loader,
                            dataset=val_data,
                            model=model,
                            output_dir=output_dir,
                            device=device,
                            rank=local_rank,
                            mode='test'
                            )


        
if __name__ == '__main__':
    main()