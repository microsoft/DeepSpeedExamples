#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=trace_lamb_64k_seq128
OUTPUT_DIR=${base_dir}/trace_bert_model_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/bert_large_lamb.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--print_steps 1 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_lamb_config_seq128.json \
--data_path_prefix /data/bert \
--jit_trace \
&> ${JOB_NAME}.log
