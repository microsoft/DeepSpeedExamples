#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
mkdir -p check/cifar/
# deepspeed --include worker-0:0 --master_port 60000 main_cifar.py      \
#     --deepspeed_config config/ds_config.json  \
#     --deepspeed   --random_ltd  \
#     --dataset cifar10vit224      \
#     --seed 1234                   \
#     --printfreq 400                \
#     --arch lvits16r224             \
#     --optimizer sgd  \
#     --lr 0.0001 --seq_len 197       \
#     --scheduler constant    \
#     --epochs 14  \
#     --batchsize 32 \
#     --data_outdir check/cifar/ | tee -a check/cifar/training.log



deepspeed --master_port 60000 main_cifar.py      \
    --deepspeed_config config/ds_config.json  \
    --deepspeed   --random_ltd  \
    --dataset cifar10vit224      \
    --seed 1234                   \
    --printfreq 400                \
    --arch vits16r224             \
    --optimizer sgd  \
    --lr 0.0001 --seq_len 197       \
    --scheduler constant    \
    --epochs 14  \
    --batchsize 128 \
    --data_outdir check/cifar/ | tee -a check/cifar/training.log