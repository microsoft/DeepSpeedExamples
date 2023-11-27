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
from torch.cuda.amp import autocast
from utils import exp_utils, loss_utils, vis_utils, train_utils

logger = logging.getLogger(__name__)


def train_epoch(config, loader, dataset, model, optimizer, scheduler, scaler,
                epoch, output_dir, device, rank, perceptual_loss, wandb_run, args):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()

    model.train()
    if config.model.backbone_fix:
        model.module.backbone.eval()
    perceptual_loss.eval()

    batch_end = time.time()
    
    for batch_idx, sample in enumerate(loader):
        iter_num = batch_idx + len(loader) * epoch
        sample = exp_utils.dict_to_cuda(sample, device)
        if config.dataset.augmentation:
            sample = train_utils.process_data(config, sample, device)
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        #torch.autograd.set_detect_anomaly(True)
        with autocast(enabled=config.train.use_amp, dtype=torch.float16):
            results = model(sample, device)
            time_meters.add_loss_value('Prediction time', time.time() - end)
            end = time.time()

            losses = loss_utils.get_losses(config, iter_num, results, sample, perceptual_loss)
            total_loss = 0.0
            for k, v in losses.items():
                if 'loss' in k:
                    total_loss += losses[k.replace('loss_', 'weight_')] * v
                    loss_meters.add_loss_value(k, v.detach().item())
            total_loss = total_loss / config.train.accumulation_step

        if config.train.use_amp:
            #with torch.autograd.detect_anomaly():
            scaler.scale(total_loss).backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Parameter: {name}, Gradient mean: {param.grad.mean().item()}, {param.grad.dtype}")
            # __import__('pdb').set_trace()
            if (batch_idx+1) % config.train.accumulation_step == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Parameter: {name}, Gradient mean: {param.grad.mean().item()}, {param.grad.dtype}")
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                scheduler.step()
        else:
            total_loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Parameter: {name}, Gradient mean: {param.grad.mean().item()}")
            # __import__('pdb').set_trace()
            if (batch_idx+1) % config.train.accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Parameter: {name}, Gradient mean: {param.grad.mean().item()}")
                # optimizer.step()
                # optimizer.zero_grad()
                # scheduler.step()
                model.backward(loss)
                model.step()
        
        time_meters.add_loss_value('Loss time', time.time() - end)
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, loss {loss_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].avg,
                recon_time=time_meters.average_meters['Prediction time'].avg,
                loss_time=time_meters.average_meters['Loss time'].avg,
                batch_time_avg=time_meters.average_meters['Batch time'].avg
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.4f} ({loss.avg:.4f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        if iter_num % config.vis_freq == 0 and rank == 0:
            if not config.dataset.train_all_frame:
                vid_clips = sample['images']
                vid_masks = sample['fg_probabilities']
            else:
                t_input = config.dataset.num_frame
                vid_clips = sample['images'][:,t_input:]
                vid_masks = sample['fg_probabilities'][:,t_input:]
            vis_utils.vis_seq(vid_clips=vid_clips,
                            vid_masks=vid_masks,
                            recon_clips=results['rgb'].reshape(vid_clips.shape),
                            recon_masks=results['mask'].reshape(vid_masks.shape),
                            iter_num=iter_num,
                            output_dir=output_dir,
                            subfolder='train_seq',
                            inv_normalize=config.train.normalize_img)

        if rank == 0:
            wandb_log = {'Train/loss': total_loss.item(),
                        'Train/lr': optimizer.param_groups[0]['lr']}
            for k, v in losses.items():
                if 'loss' in k:
                    wandb_log['Train/{}'.format(k)] = v.item()
            wandb_run.append(wandb_log)
            torch.save(wandb_run,f'wandb_run_{args.name}.pt')
        dist.barrier()
        batch_end = time.time()
    
    del losses, total_loss, results, sample

