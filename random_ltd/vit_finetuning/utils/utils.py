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

import collections
import torch
from torch import Tensor
import torch.nn as nn
import torch.backends.cudnn as cudnn
import subprocess
import os
import time
import shutil
from datetime import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
sys.path.append("..")
from third_party import models
import numpy as np
import torch.nn as nn
import math

def run_cmd(cmd_str, prev_sp=None):
  """
  This function runs the linux command cmr_str as a subprocess after waiting
  for prev_sp subprocess to finish
  """
  if prev_sp is not None:
    prev_sp.wait()
  return subprocess.Popen(cmd_str, shell=True)#, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))


def get_model(model_name, nchannels=3, imsize=32, nclasses=10, half=False):

  ngpus = torch.cuda.device_count()

  print("=> creating model '{}'".format(model_name))
  if imsize < 128 and model_name in models.__dict__:
    model = models.__dict__[model_name](num_classes=nclasses, nchannels=nchannels)
  else:
    model = models.__dict__[model_name](num_classes=nclasses, nchannels=nchannels)
  model = nn.DataParallel(model).cuda()
  cudnn.benchmark = True
  if half:
    print('Using half precision except in Batch Normalization!')
    model = model.half()
    for module in model.modules():
      if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):
        module.float()
  return model

def get_optimizer(optimizer_name, parameters, lr, momentum=0, weight_decay=0):
  if optimizer_name == 'sgd':
    return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
  elif optimizer_name == 'nesterov_sgd':
    return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
  elif optimizer_name == 'rmsprop':
    return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
  elif optimizer_name == 'adagrad':
    return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif optimizer_name == 'adam':
    return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

def get_scheduler(scheduler_name, optimizer, num_epochs, **kwargs):
  if scheduler_name == 'constant':
    return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
  elif scheduler_name == 'step2':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 2), gamma=0.1, **kwargs)
  elif scheduler_name == 'step3':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 3), gamma=0.1, **kwargs)
  elif scheduler_name == 'exponential':
    return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
  elif scheduler_name == 'cosine':
    return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)
  elif scheduler_name == 'step-more':
    return lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, **kwargs)

def run_cmd(cmd_str, prev_sp=None):
  """
  This function runs the linux command cmr_str as a subprocess after waiting
  for prev_sp subprocess to finish
  """
  if prev_sp is not None:
    prev_sp.wait()
  return subprocess.Popen(cmd_str, shell=True)#, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))


class LossTracker(object):
  def __init__(self, num, prefix='', print_freq=1):
    self.print_freq=print_freq
    self.batch_time = AverageMeter('Time', ':6.3f')
    self.losses = AverageMeter('Loss', ':.4e')
    self.top1 = AverageMeter('Acc@1', ':6.2f')
    self.top5 = AverageMeter('Acc@5', ':6.2f')
    self.progress = ProgressMeter( num, [self.batch_time, self.losses, self.top1, self.top5], prefix=prefix)
    self.end = time.time()

  def update(self, loss, output, target):
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    self.losses.update(loss.item(), output.size(0))
    self.top1.update(acc1[0], output.size(0))
    self.top5.update(acc5[0], output.size(0))

  def display(self, step):
    self.batch_time.update(time.time() - self.end)
    self.end = time.time()
    if step % self.print_freq == 0:
      self.progress.display(step)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def balance_order(order, dataset, num_classes=10):
    size_each_class = len(dataset) // num_classes
   
    class_orders = collections.defaultdict(list)
    for i in range(len(order)):
        class_orders[dataset.targets[order[i]]].append(i)
    # take each group containing the next easiest image for each class,
    # and putting them according to diffuclt-level in the new order
    length = []
    for cls in range(num_classes):
        length.append(len(class_orders[cls]))
    print ("minmax", min(length), max(length))  
    new_order = []
    
    for group_idx in range(min(length)):            
        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes)])
        new_order.extend([order[idx] for idx in group])
        
    for group_idx in range(min(length), max(length)):
        cls_idx = [cls for cls in range(num_classes) if group_idx<length[cls]]
        group = sorted([class_orders[cls][group_idx] for cls in cls_idx])
        new_order.extend([order[idx] for idx in group])        
    assert len(new_order) == len(order)
    return new_order


def balance_order_val(order, dataset, num_classes=10,valp=0.1):
    size_each_class = len(dataset) // num_classes
    print (" size_each_class ", size_each_class )
    class_orders = collections.defaultdict(list)
    for i in range(len(order)):
        class_orders[dataset.targets[order[i]]].append(i)
    # take each group containing the next easiest image for each class,
    # and putting them according to diffuclt-level in the new order
    length = []
    new_order_val = []
    class_orders_new = collections.defaultdict(list)
    for cls in range(num_classes):
        np.random.seed(cls)
        tmp_id = np.array(class_orders[cls])
        random_id = np.random.choice(len(tmp_id),int(len(tmp_id)*valp),replace=False)
        tmp_id_val = [np.array(tmp_id)[ID] for ID in random_id ]  
        new_order_val.extend([order[idx] for idx in tmp_id_val ])
        
        class_orders_new[cls].extend([x for x in class_orders[cls] if x not in tmp_id_val])
        length.append(len(class_orders_new[cls]))
        
    new_order = []
    for group_idx in range(min(length)):            
        group = sorted([class_orders_new[cls][group_idx] for cls in range(num_classes)])
        new_order.extend([order[idx] for idx in group])
        
    for group_idx in range(min(length), max(length)):
        cls_idx = [cls for cls in range(num_classes) if group_idx<length[cls]]
        group = sorted([class_orders_new[cls][group_idx] for cls in cls_idx])
        new_order.extend([order[idx] for idx in group])  
           
    assert np.sum(new_order)+np.sum(new_order_val) == np.sum(order)
    
    return new_order,new_order_val

        
def get_pacing_function(total_step, total_data, args):
    """Return a  pacing function  w.r.t. step.
    input:
    a:[0,large-value] percentage of total step when reaching to the full data. This is an ending point (a*total_step, total_data)) 
    b:[0,1]  percentatge of total data at the begining of the training. Thia is a starting point (0,b*total_data))
    """
    a = args.pacing_a
    b = args.pacing_b 
    index_start = b*total_data
    if args.pacing_f == 'linear':
      rate = (total_data - index_start)/(a*total_step)
      def _linear_function(step):
        return int(rate *step + index_start)
      return _linear_function
    
    elif args.pacing_f == 'quad':
      rate = (total_data-index_start)/(a*total_step)**2  
      def _quad_function(step):
        return int(rate*step**2 + index_start)
      return _quad_function
    
    elif args.pacing_f == 'root':
      rate = (total_data-index_start)/(a*total_step)**0.5
      def _root_function(step):
        return int(rate *step**0.5 + index_start)
      return _root_function
    
    elif args.pacing_f == 'step':
      threshold = a*total_step
      def _step_function(step):
        return int( total_data*(step//threshold) +index_start)
      return _step_function      

    elif args.pacing_f == 'exp':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      rate =  (total_data-tilde_b)/(np.exp(c)-1)
      constant = c/tilde_a
      def _exp_function(step):
        if not np.isinf(np.exp(step *constant)):
            return int(rate*(np.exp(step*constant)-1) + tilde_b )
        else:
            return total_data
      return _exp_function

    elif args.pacing_f == 'log':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      ec = np.exp(-c)
      N_b = (total_data-tilde_b)
      def _log_function(step):
        return int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b )
      return _log_function



def get_seq_function(total_step, total_data, args):
    """Return a  pacing function  w.r.t. step.
    input:
    a:[0,large-value] percentage of total step when reaching to the full data. This is an ending point (a*total_step, total_data)) 
    b:[0,1]  percentage of total data at the beginning of the training. This is a starting point (0,b*total_data))
    """
    a = args.seq_func_reach_steps
    b = args.seq_func_init_length 
    index_start = b*total_data
    if args.drop_token_type == 'linear':
      rate = (total_data - index_start)/(a*total_step)
      def _linear_function(step):
        return int(rate *step + index_start)
      return _linear_function
    
    elif args.drop_token_type == 'quad':
      rate = (total_data-index_start)/(a*total_step)**2  
      def _quad_function(step):
        return int(rate*step**2 + index_start)
      return _quad_function
    
    elif args.drop_token_type == 'root':
      rate = (total_data-index_start)/(a*total_step)**0.5
      def _root_function(step):
        return int(rate *step**0.5 + index_start)
      return _root_function
    
    elif args.drop_token_type == 'step':
      threshold = a*total_step
      def _step_function(step):
        return int( total_data*(step//threshold) +index_start)
      return _step_function      

    elif args.drop_token_type == 'exp':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      rate =  (total_data-tilde_b)/(np.exp(c)-1)
      constant = c/tilde_a
      def _exp_function(step):
        if not np.isinf(np.exp(step *constant)):
            return int(rate*(np.exp(step*constant)-1) + tilde_b )
        else:
            return total_data
      return _exp_function

    elif args.drop_token_type == 'log':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      ec = np.exp(-c)
      N_b = (total_data-tilde_b)
      def _log_function(step):
        return int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b )
      return _log_function



class TokenAnnealingLR(object):
    """Anneals the learning rate."""
    def __init__(self, optimizer, max_lr, min_lr, decay_style,
                  lr_decay_tokens=None,
                  lr_warmup_tokens=None):
        # Class values.
        self.optimizer = optimizer
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        self.num_steps = 0
        self.decay_tokens = lr_decay_tokens
        self.num_tokens = 0
        self.warmup_tokens = lr_warmup_tokens
        self.decay_style = decay_style
        # Set the learning rate
        self.step(0)
        print('> learning rate decay style: {}'.format(self.decay_style))
        
    def get_lr(self):
        # reserved_factor = math.sqrt(reserved_length / 2048.0)
        reserved_factor = 1.0
        # Use linear warmup for the initial part.

        if self.warmup_tokens > 0 and self.num_tokens <= self.warmup_tokens:
            return self.max_lr * float(self.num_tokens) / \
                float(self.warmup_tokens) * reserved_factor
        # If the learning rate is constant, just return the initial value.
        if self.decay_style == 'constant':
            return self.max_lr
        # token-based decay
        if self.num_tokens > self.decay_tokens:
            return self.min_lr
        num_tokens_ = self.num_tokens - self.warmup_tokens
        decay_tokens_ = self.decay_tokens - self.warmup_tokens
        decay_ratio = float(num_tokens_) / float(decay_tokens_)

        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = self.max_lr - self.min_lr
        if self.decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.decay_style))
       
        return (self.min_lr + coeff * delta_lr) * reserved_factor
    def step(self, token_num=None):
        self.num_tokens = token_num
        self.num_steps += 1
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

class LayerTokenAnnealingLR(object):
    """Anneals the learning rate."""
    def __init__(self, optimizer, max_lr, min_lr, decay_style,
                  total_consumed_token_layers=None,
                  lr_warmup_tokens=None,
                  depth=None):
        # Class values.
        self.optimizer = optimizer
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        self.num_steps = 0

        self.num_tokens = 0
        self.warmup_tokens = lr_warmup_tokens
        self.decay_style = decay_style
    
        self.warmup_tokens = lr_warmup_tokens * depth # make it to token-layers
        self.decay_tokens = total_consumed_token_layers - self.warmup_tokens
        # Set the learning rate
        self.step(0,self.num_tokens)
        print('> learning rate decay style: {}'.format(self.decay_style))
        
    def get_lr(self):
        # reserved_factor = math.sqrt(reserved_length / 2048.0)
        reserved_factor = 1.0
        # Use linear warmup for the initial part.

        if self.warmup_tokens > 0 and self.num_tokens <= self.warmup_tokens:
            return self.max_lr * float(self.num_tokens) / \
                float(self.warmup_tokens) * reserved_factor
        # If the learning rate is constant, just return the initial value.
        if self.decay_style == 'constant':
            return self.max_lr
        # token-based decay
        if self.num_tokens > self.decay_tokens:
            return self.min_lr
        num_tokens_ = self.num_tokens - self.warmup_tokens
        decay_tokens_ = self.decay_tokens - self.warmup_tokens
        decay_ratio = float(num_tokens_) / float(decay_tokens_)

        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = self.max_lr - self.min_lr
        if self.decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.decay_style))
       
        return (self.min_lr + coeff * delta_lr) * reserved_factor

    def step(self, increment, token_num=None):
        self.num_tokens = token_num
        self.num_steps += increment
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
