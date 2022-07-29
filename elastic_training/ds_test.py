import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.multiprocessing as mp
import os

from typing import Any, Dict, Optional, Tuple
import deepspeed
from torch.distributed.elastic.metrics.api import prof
import signal
from datetime import timedelta
# from train import trainer
# In[29]:
import torch.distributed as dist
# import deepspeed
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    SimpleElasticAgent,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.multiprocessing import PContext, start_processes
import shutil
from torch.distributed.elastic.utils.logging import get_logger


transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

model = torchvision.models.resnet152()

def trainer(model,trainloader):
    def usr_signal(signalNumber, frame):
        print("Recieved usr signal")
        dist.destroy_process_group()
        print("Group Destroyed")
        dist.init_process_group(
            backend='nccl', init_method="env://", timeout=timedelta(seconds=20) 
            )
    def SIGABRT_signal(signalNumber, frame):
        print("Recieved abort")

    signal.signal(signal.SIGUSR1, usr_signal)
    # signal.signal(signal.SIGABRT, SIGABRT_signal)


    device_id = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend='nccl', init_method="env://", timeout=timedelta(seconds=20) 
    )


    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100

    print("Init dev:",device_id)
    
    model.to(device_id)
    # model = DistributedDataParallel(model, device_ids=[device_id])
    # log_dist("Creating DeepSpeed engine", ranks=[0], level=logging.INFO)
    ds_config = {
        "train_micro_batch_size_per_gpu": 128,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "fp16": {
            "enabled": False
        }
    }

    model, _, _, _ = deepspeed.initialize(model=model,
                                          model_parameters=model.parameters(),
                                          config=ds_config,
                                          dist_init_required=False
                                          )
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Construct data_loader, optimizer, etc.

    for epoch in range(epochs):
        # try:
        for data, labels in trainloader:
            data = data.to(device_id)
            labels = labels.to(device_id)
            # optimizer.zero_grad()
            loss = criterion(model(data), labels)
            print("Before backward:",device_id)
            model.backward(loss)
            model.step()
            print("dev:",device_id, loss.item())

trainer(model,trainloader)