#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python main_cifar.py      \
    --deepspeed_config config/ds_config.json  \
    --deepspeed   --random_ltd           \
    --dataset cifar10vit224      \
    --seed 1234                   \
    --printfreq 400                \
    --arch lvits16r224             \
    --optimizer sgd  \
    --lr 0.0001 --seq_len 197       \
    --scheduler constant             \
    --epochs 14  \
    --data_outdir check_correct/test-sep15 | tee -a check_correct/test-sep15/training.log