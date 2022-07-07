#### Install

``pip install -r requirement.txt``

You will also need to install DeepSpeed *staging_compression_library_v1* https://github.com/microsoft/DeepSpeed-internal/tree/staging_compression_library_v1

#### Key File: train.py

The python code is modified based on (https://github.com/microsoft/DeepSpeedExamples-internal/tree/master/cifar). The key added feature is the compression pipeline

#### Folders (config)

* **config:** This folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.

#### bash script 
* **run_compress.sh**  This bash script contains jobs for training a checkpoint and then compressing this checkpoint. 
