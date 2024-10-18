# Domino Project Setup Guide

This guide provides instructions to set up a local Docker environment and configure Python, PyTorch, and additional tools for the Domino project

## Local (docker) environment setup
To start a Docker container for the Domino environment, run the following commands:
```
docker run -d -t --network=host --gpus all --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name domino -v /etc/localtime:/etc/localtime nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
docker exec -it domino bash
```

## Set Up Python Environment
Enter the Docker container and set up the Python environment:
```
# Enter container
apt update
apt upgrade -y
apt install -y wget git vim

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts on the installer screens.
source /root/.bashrc

# Create a Python environment for Domino
conda create -n domino python==3.10
conda activate domino

# Create and navigate to the working directory
mdkir /workspace/code
cd /workspace/code
```

## Set up Pytorch environment

You can install PyTorch from conda or compile it from source for better performance.

### Install from Conda (Recommended)
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
 
### Compile PyTorch from Source (Optional)
```
git clone -b v2.1.0 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
conda install intel::mkl-static intel::mkl-include
conda install -c pytorch magma-cuda121 # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop

# Build torchvision
cd ..
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py develop
```

## Build Apex
To build NVIDIA Apex from source:
```
cd /workspace/code
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
```

## Install Additional Libraries
```
Clone Domino Repositories
```

## Clone Domino Repositories
Set up an SSH key for GitHub following the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?platform=linux).

After configuring SSH, clone the required repositories:
```
cd /workspace/code
git clone git@github.com:zhangsmallshark/DeepSpeedExamples.git
git clone git@github.com:zhangsmallshark/DeepSpeed-internal.git
```

## Build DeepSpeed
```
cd /workspace/code/DeepSpeed-internal
pip install -e .
```

## Prepare the Dataset
Follow the instructions from [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing#download-and-pre-process-training-dataset) to prepare the training dataset:
```
git clone https://github.com/microsoft/Megatron-DeepSpeed.git

mkdir /workspace/dataset
cd /workspace/dataset
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz
python /workspace/code/Megatron-DeepSpeed/tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix my-gpt2 \
    --vocab-file gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

## Execute Domino Training
```
cd /workspace/code/DeepSpeedExamples/training/Deepspeed-Domino
# Adjust GPU number, dataset path, and training configuration in pretrain_gpt.sh if needed.
bash pretrain_gpt.sh
```