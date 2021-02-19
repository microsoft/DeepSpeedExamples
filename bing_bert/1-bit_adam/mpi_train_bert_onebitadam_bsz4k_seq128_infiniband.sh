#!/bin/bash

# Note: Please use the deepspeed launch script for most cases.
# For advanced users, mpirun or other MPI launchers can be used
# with this script as follows.
# mpirun -n 2 [launcher-args] python ${base_dir}/../deepspeed_train.py
# As an example, below we include the command we used

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_adam_4k_seq128_mpirun_infiniband
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

# LD_PRELOAD is used to load a specific nccl version. Remove it if needed.
mpirun -n 128 -ppn 8 -f /tmp/deepspeed_mvapich_hostfile -env MV2_SMP_USE_CMA=0 -env MV2_DEBUG_SHOW_BACKTRACE=1 -env MV2_USE_CUDA=1 -env MV2_SUPPORT_DL=1 -env MV2_ENABLE_AFFINITY=0 -env MV2_INTER_ALLGATHER_TUNING=5 -env MV2_CUDA_USE_NAIVE=0 -env NCCL_TREE_THRESHOLD=0 -env PYTHONUNBUFFERED=True -env MV2_USE_GDRCOPY=0 -env MV2_USE_GDR=0 -env NCCL_VERSION=2.8.3 -env PYTHON_VERSION=3 -env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3 python ${base_dir}/../deepspeed_train.py \
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
--deepspeed_config ${base_dir}/deepspeed_bsz4k_onebitadam_config_seq128_mpi_infiniband.json \
--data_path_prefix /data/bert \
&> ${JOB_NAME}.log
