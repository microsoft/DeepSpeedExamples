# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import time
import warnings
import json
import collections
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Subset

from utils import get_dataset, get_model, get_optimizer, get_scheduler, get_seq_function
from utils import  LossTracker, run_cmd
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data-dir', default='dataset',
                    help='path to dataset')
parser.add_argument('--order-dir', default='cifar10-cscores-orig-order.npz',
                    help='path to train val idx')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model sitecture: (default: resnet18)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--printfreq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=128, type=int,
                    help='mini-batch size (default: 256), this is the total')
parser.add_argument('--optimizer', default="sgd", type=str,
                    help='optimizer')
parser.add_argument('--scheduler', default="cosine", type=str,
                    help='lr scheduler')
parser.add_argument("--token_scheduler", action="store_true", help="use token scheduler or not.")
parser.add_argument("--warmup", action="store_true", help="use token scheduler or not.")
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--half', default=False, action='store_true',
                    help='training with half precision')
# curriculum params
parser.add_argument("--data_outdir", default=".", type=str, help="output directory")
parser.add_argument('--seq_len', default=197, type=int, help='image sequence length')
args = parser.parse_args()
def main():
    set_seed(args.seed) 
    # create training and validation datasets and intiate the dataloaders
    tr_set = get_dataset(args.dataset, args.data_dir, 'train',rand_fraction=args.rand_fraction)
    val_set = get_dataset(args.dataset, args.data_dir, 'val')    
        
    train_loader = torch.utils.data.DataLoader(tr_set, batch_size=args.batchsize,\
                              shuffle=True, num_workers=args.workers, pin_memory=True)  
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batchsize*2,
                      shuffle=False, num_workers=args.workers, pin_memory=True)


    model = get_model(args.arch, tr_set.nchannels, tr_set.imsize, len(tr_set.classes), args.half)
    print (model)
    #initial training

    total_iteration = args.epochs*len(train_loader)+1
    total_batch_size = args.batchsize
    num_update_steps_per_epoch = len(train_loader)
    TRAIN_TOKENS = total_batch_size * args.epochs * num_update_steps_per_epoch * args.seq_len 
    if args.warmup:  
        lr_warmup_tokens = math.ceil(num_update_steps_per_epoch*total_batch_size*args.seq_len)
    else:
        lr_warmup_tokens = 0
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=total_iteration)


    iterations = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "iter": [0,] }
    reserved_length = 0
    criterion = nn.CrossEntropyLoss().cuda()
    seq_function = None
    if args.seq_func_reach_steps!=0:
        seq_function = get_seq_function(total_iteration, args.seq_len, args)
        print ('using seq function')
    else:
        print ("no using seq function", args.seq_func_reach_steps )

    val_loss, val_acc1 = validate(val_loader, model, criterion)
    print (f'{-1} epoch at time {args.drop_token_type} {reserved_length}')
    print (" %s  and val accurarcy %s, and used_token %s"%(iterations,  val_acc1.item()))     
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc1.item())  
    for epoch in range(args.epochs): 
        start_time = time.time()
        tr_loss, tr_acc1, iterations = train(seq_function, train_loader, model, criterion, optimizer, scheduler, epoch, iterations, reserved_length, consumed_train_tokens)
        val_loss, val_acc1 = validate(val_loader, model, criterion)
        time_epoch = time.time() - start_time
        print (f'{epoch} epoch at time {time_epoch}s')
        print ("%s epoch %s iter w/ LR %s  and val accurarcy %s, and used_token %s"%(epoch, iterations,scheduler.get_lr(), val_acc1.item(), consumed_train_tokens))           
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc1.item())  
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc1.item())
        history["iter"].append(iterations)
        torch.save(history,f"{args.data_outdir}/stat.pt")  
        
def train(seq_function, train_loader, model, criterion, optimizer,scheduler, epoch, iterations):
  # switch to train mode
  model.train()
  tracker = LossTracker(len(train_loader), f'Epoch: [{epoch}]', args.printfreq)
  for i, (images, target) in enumerate(train_loader):
    iterations += 1
    images, target = cuda_transfer(images, target)
    output = model(images)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    tracker.update(loss, output, target)
    tracker.display(i)

  return tracker.losses.avg, tracker.top1.avg,  iterations

def validate(val_loader, model, criterion):
  # switch to evaluate mode
  model.eval()
  with torch.no_grad():
    tracker = LossTracker(len(val_loader), f'val', args.printfreq)
    for i, (images, target) in enumerate(val_loader):
      images, target = cuda_transfer(images, target)
      output = model(images)
      loss = criterion(output, target)
      tracker.update(loss, output, target)
      tracker.display(i)
  return tracker.losses.avg, tracker.top1.avg

def set_seed(seed=None):
    if seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

def cuda_transfer(images, target):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    if args.half: images = images.half()
    return images, target

if __name__ == '__main__':
    main()

