#!/bin/bash

# Note: Please use the deepspeed launch script for most cases.
# For advanced users, mpirun or other MPI launchers can be used
# with this script as follows.
# mpirun -n 2 [launcher-args] python ${base_dir}/../deepspeed_train.py
# As an example, below we include the command we used

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_adam_4k_seq128_mpirun_ethernet
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

# LD_PRELOAD is used to load a specific nccl version. Remove it if needed.
# NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 are used to disable infiniband. Remove it if needed.
mpirun -n 128 -npernode 4 -hostfile /job/hostfile -x UCX_TLS=tcp --mca btl ^openib --mca btl_tcp_if_include eth0 -x NCCL_TREE_THRESHOLD=0 -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=eth0 -x LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3 python ${base_dir}/../deepspeed_train.py \
--cf ${base_dir}/../bert_large.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed_mpi \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 40 \
--lr_schedule "LE" \
--lr_offset 0.0 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz4k_onebitadam_config_seq128_mpi_ethernet.json \
--data_path_prefix /data/bert \
&> ${JOB_NAME}.log
