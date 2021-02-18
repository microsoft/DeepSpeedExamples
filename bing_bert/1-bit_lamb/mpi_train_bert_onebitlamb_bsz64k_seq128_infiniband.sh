#!/bin/bash

# Note: Please use the deepspeed launch script for most cases.
# For advanced users, mpirun or other MPI launchers can be used
# with this script as follows.
# mpirun -n 2 [launcher-args] python ${base_dir}/../deepspeed_train.py
# As an example, below we include the command we used

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_lamb_64k_seq128_mpirun_infiniband
OUTPUT_DIR=${base_dir}/bert_model_outputs

mkdir -p $OUTPUT_DIR

mpirun -n 128 -ppn 8 -f /tmp/deepspeed_mvapich_hostfile -env MV2_SUPPORT_DL=1 python ${base_dir}/../deepspeed_train.py \
--cf ${base_dir}/../bert_large_lamb.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed_mpi \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 40 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz64k_onebitlamb_config_seq128_mpi_infiniband.json \
--data_path_prefix /data/bert \
&> ${JOB_NAME}.log
