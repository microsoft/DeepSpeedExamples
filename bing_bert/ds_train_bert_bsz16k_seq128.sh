#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lamb_64k_seq128_64GPUs_2
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--print_steps 100 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz16K_lamb_config.json \
--data_path_prefix /data/bert \
&> 128_lamb_64GPU_80_2.log
