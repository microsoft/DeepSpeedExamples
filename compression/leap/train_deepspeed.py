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

os.environ["WANDB__SERVICE_WAIT"] = "300"

from config.config import config, update_config
from model.model import LEAP
from model.external.perceptual_loss import VGGPerceptualLoss
from utils import dist_utils, exp_utils, train_utils
from scripts.trainer import train_epoch
from scripts.evaluator import evaluation
import deepspeed 


def parse_args():
    parser = argparse.ArgumentParser(description='Train LEAP')
    parser.add_argument(
        '--name', help='experiment  name', required=True, type=str)
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser = deepspeed.add_config_arguments(parser)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

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
    deepspeed.init_distributed()
    
   # set wandb
    wandb_run = []
    if local_rank == 0:
        wandb_name = config.exp_name
        wandb_proj_name = config.exp_group
        #wandb_run = wandb.init(project=wandb_proj_name, group=wandb_name)
        wandb_run.append({
            "exp_name": config.exp_name,
            "batch_size": config.train.batch_size,
            "total_iteration": config.train.total_iteration,
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
        })
    else:
        wandb_run = None
    torch.distributed.barrier()
    # get model
    model = LEAP(config).to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)

    # get optimizer
    optimizer = train_utils.get_optimizer(config, model)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.train.schedular_warmup_iter,
                                                             num_training_steps=config.train.total_iteration)
    scaler = torch.cuda.amp.GradScaler()

    # load pre-trained model
    if len(config.train.pretrain_path) > 0:
        model = train_utils.load_pretrain(config, model, config.train.pretrain_path)
    
    # resume training
    best_psnr = 0.0
    ep_resume = None
    if config.train.resume:
        model, optimizer, scheduler, scaler, ep_resume, best_psnr = train_utils.resume_training(
                                                                        model, optimizer, scheduler, scaler,
                                                                        output_dir, cpt_name='cpt_last.pth.tar')
        print('LR after resume {}'.format(optimizer.param_groups[0]['lr']))
    else:
        print('No resume training')

    # distributed training
    ddp = False
    model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        find_unused = True if (not config.model.backbone_fix) else False
        
        model, optimizer, _, scheduler = deepspeed.initialize(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    lr_scheduler=scheduler,
                    dist_init_required=True)        
        
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=find_unused)
        device_num = len(device_ids)
        ddp = True

    # get dataset and dataloader
    train_data = train_utils.get_dataset(config, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.train.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=True,
                                               sampler=train_sampler)
    val_data = train_utils.get_dataset(config, split='val')
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=config.test.batch_size, 
                                             shuffle=False,
                                             num_workers=int(config.workers), 
                                             pin_memory=True, 
                                             drop_last=False)
    
    start_ep = ep_resume if ep_resume is not None else 0
    end_ep = int(config.train.total_iteration / len(train_loader)) + 1
    
    if config.dataset.name == 'omniobject3d':
        val_interval = 15
    elif config.dataset.name == 'kubric':
        val_interval = 5
    elif config.dataset.name == 'objaverse':
        val_interval = 1
    elif config.dataset.name == 'dtu':
        val_interval = 2
        
    # train
    for epoch in range(start_ep, end_ep):
        train_sampler.set_epoch(epoch)
        train_epoch(config,
                    loader=train_loader,
                    dataset=train_data,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    output_dir=output_dir,
                    device=device,
                    rank=local_rank,
                    perceptual_loss=perceptual_loss,
                    wandb_run=wandb_run,
                    args = args
                    )
        
        if local_rank == 0:
            train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'schedular': scheduler.state_dict(),
                    }, 
                    checkpoint=output_dir, filename="cpt_last.pth.tar")
            
        if (epoch % val_interval == 0) and (epoch > 0):
            #print('Doing validation')
            cur_psnr, return_dict = evaluation(config,
                                               loader=val_loader,
                                               dataset=val_data,
                                               model=model,
                                               epoch=epoch,
                                               output_dir=output_dir,
                                               device=device,
                                               rank=local_rank,
                                               wandb_run=wandb_run,
                                               mode='val'
                                               )
            
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                if local_rank == 0:
                    train_utils.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'schedular': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'best_psnr': best_psnr,
                            'eval_dict': return_dict,
                        }, 
                    checkpoint=output_dir, filename="cpt_best_psnr_{}.pth.tar".format(best_psnr))
            
            if local_rank == 0:
                logger.info('Best PSNR {}, current PSNR {}'.format(best_psnr, cur_psnr))
        
        dist.barrier()
        torch.cuda.empty_cache()

        
if __name__ == '__main__':
    main()
