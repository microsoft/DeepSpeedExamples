import os
import sys
import json
import time
import datetime
import warnings
import torch
from easydict import EasyDict as edict
from pathlib import Path
import logging

def progress_bar(msg=None):
    L = []
    if msg:
        L.append(msg)

    msg = ''.join(L)
    sys.stdout.write(msg+'\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


class AverageMeters:
    def __init__(self):
        super().__init__()
        self.average_meters = {}

    def add_loss_value(self, loss_name, loss_val, n=1):
        if loss_name not in self.average_meters:
            self.average_meters[loss_name] = AverageMeter()
        self.average_meters[loss_name].update(loss_val, n=n)


class Monitor:
    def __init__(self, hosting_file):
        self.hosting_file = hosting_file

    def log_train(self, epoch, errors):
        log_errors(epoch, errors, self.hosting_file)


def log_errors(epoch, errors, log_path=None):
    now = time.strftime("%c")
    message = "(epoch: {epoch}, time: {t})".format(epoch=epoch, t=now)
    for k, v in errors.items():
        message = message + ",{name}:{err}".format(name=k, err=v)

    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
    return message


def print_args(config):
    print("======= Options ========")
    edict_type = type(edict())
    for k, v in sorted(config.items()):
        if type(v) == edict_type:
            print(k + ':')
            for k2, v2 in v.items():
                print("     {}: {}".format(k2, v2))
        else:
            print("{}: {}".format(k, v))
    print("========================")


def save_args(config_file_path, save_folder, save_name):
    if os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)
    from shutil import copyfile
    copyfile(config_file_path, save_path)


def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))
    root_dir = (this_dir / '..').resolve()
    output_dir = (root_dir / cfg.output_dir).resolve()
    log_dir = (root_dir / cfg.log_dir).resolve()
    if not output_dir.exists():
        print('Creating output dir {}'.format(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
    if not log_dir.exists():
        print('Creating log dir {}'.format(log_dir))
        log_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.dataset.name
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    # if cfg.dataset.category is not None:
    #     cfg_name += '_' + cfg.dataset.category
    exp_name = cfg.exp_name

    final_output_dir = output_dir / dataset_name / cfg_name / exp_name
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=str(final_log_file),
                        format=head,
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    #from IPython import embed; embed()

    tb_log_dir = log_dir / dataset_name / cfg_name / (exp_name + time_str)
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tb_log_dir)


def dict_to_cuda(batch, device):
    return {k: try_to_cuda(v, device) for k, v in batch.items()}

def try_to_cuda(t, device):
    try:
        t = t.to(device)
    except AttributeError:
        pass
    return t