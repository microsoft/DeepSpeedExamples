import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import resnet_model
import os

SUMMARY_WRITER_DIR_NAME = 'runs'


def get_sample_writer(name, base=".."):
    """Returns a tensorboard summary writer
    """
    return SummaryWriter(
        log_dir=os.path.join(base, SUMMARY_WRITER_DIR_NAME, name))


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-lr',
                        '--learning_rate',
                        default=5e-3,
                        type=float,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument(
        '--job_name',
        type=str,
        default=None,
        help="This is the path to store the output and TensorBoard results.")

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


deepspeed.init_distributed()


def master_process(args):
    return dist.get_rank() == 0


########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

args = add_argument()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if torch.distributed.get_rank() != 0:
    # might be downloading cifar data, let rank 0 download first
    torch.distributed.barrier()
trainset = torchvision.datasets.CIFAR100(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=16)

testset = torchvision.datasets.CIFAR100(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1024,
                                         shuffle=False,
                                         num_workers=16)
if torch.distributed.get_rank() == 0:
    # cifar data is downloaded, indicate other ranks can proceed
    torch.distributed.barrier()

classes = ['label {}'.format(x) for x in range(100)]

net = resnet_model.PreActResNet50()
parameters, names = [], []
for n, p in list(net.named_parameters()):
    if p.requires_grad:
        parameters.append(p)
        names.append(n)
print('num parameter: ', len(names), len(parameters))
print('parameter names: ', names)
optimizer_grouped_parameters = [{'params': parameters, 'no_freeze': False}]

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=net,
    model_parameters=optimizer_grouped_parameters,
    training_data=trainset)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

if master_process(args):
    summary_writer = get_sample_writer(name=args.job_name, base='./output/')
global_step = 0
global_data_samples = 0


def update_lr_this_step():
    global global_step
    lr_offset = 0
    warmup_proportion = 0.1
    learning_rate = args.learning_rate
    decay_rate = 0.9
    decay_step = 250
    total_training_steps = 10000
    degree = 2.0

    x = global_step / total_training_steps
    warmup_end = warmup_proportion * total_training_steps
    if x < warmup_proportion:
        # lr_this_step = (x / warmup_proportion)**degree
        lr_this_step = x / warmup_proportion
    else:
        # lr_this_step = decay_rate**((global_step - warmup_end) / decay_step)
        lr_this_step = 1
    lr_this_step = lr_this_step * learning_rate
    lr_this_step += lr_offset
    return lr_this_step


for epoch in range(args.epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    model_engine.train()
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)

        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        global_data_samples += (model_engine.train_micro_batch_size_per_gpu() *
                                dist.get_world_size())

        model_engine.backward(loss)

        if model_engine.is_gradient_accumulation_boundary():
            global_step += 1
            lr_this_step = update_lr_this_step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            model_engine.step()
            # print statistics
            if master_process(args):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                summary_writer.add_scalar(f'Train/step/train_loss',
                                          loss.item(), global_step)
                summary_writer.add_scalar(f'Train/sample/train_loss',
                                          loss.item(), global_data_samples)
                summary_writer.add_scalar(f'Train/sample/lr', lr_this_step,
                                          global_data_samples)
                lamb_coeffs = optimizer.get_lamb_coeffs()
                if len(lamb_coeffs) > 0:
                    assert len(lamb_coeffs) == len(names)
                    for i in range(len(lamb_coeffs)):
                        summary_writer.add_scalar(
                            'Lamb/step/coeff_{}_{}'.format(i, names[i]),
                            lamb_coeffs[i], global_step)
                        # summary_writer.add_scalar('Lamb/sample/coeff_{}_{}'.format(i, names[i]), lamb_coeffs[i], global_data_samples)
        else:
            model_engine.step()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(model_engine.local_rank))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(
                model_engine.local_rank)).sum().item()
    if master_process(args):
        summary_writer.add_scalar(f'Test/epoch/test_accuracy',
                                  float(correct) / total, epoch + 1)
        summary_writer.add_scalar(f'Test/sample/test_accuracy',
                                  float(correct) / total, global_data_samples)

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(model_engine.local_rank))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(
            model_engine.local_rank)).sum().item()
if master_process(args):
    print('Accuracy of the network on the test images: %d %%' %
          (100 * correct / total))
