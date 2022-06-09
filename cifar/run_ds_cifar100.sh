#!/bin/bash

mkdir -p output

JOB_NAME=cifar100_onebitlamb_bsz1k_lr1e-2_0p3_0p01_freeze1000
deepspeed cifar100_deepspeed.py --deepspeed --deepspeed_config ds_config_onebitlamb.json --epochs 250 --batch_size 16 --learning_rate 1e-2 --job_name $JOB_NAME > output/$JOB_NAME.log

JOB_NAME=cifar100_lamb_bsz1k_lr1e-2_0p3_0p01
deepspeed cifar100_deepspeed.py --deepspeed --deepspeed_config ds_config_lamb.json --epochs 250 --batch_size 16 --learning_rate 1e-2 --job_name $JOB_NAME > output/$JOB_NAME.log
