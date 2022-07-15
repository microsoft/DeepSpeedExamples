#### Install

``pip install torch torchvision``
You will also need to install updated DeepSpeed verion, which contains the compression libarary

#### Key File: train.py

The python code is modified based on (https://github.com/microsoft/DeepSpeedExamples/tree/master/cifar). The key added feature is the compression pipeline

#### Folders (config)

* **config:** This folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.

#### bash script 
* **run_compress.sh**  This bash script contains jobs for training a checkpoint and then compressing this checkpoint. 
