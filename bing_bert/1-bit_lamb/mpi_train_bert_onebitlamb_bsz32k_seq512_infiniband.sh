#!/bin/bash

# Note: Please use the deepspeed launch script for most cases.
# For advanced users, mpirun or other MPI launchers can be used
# with this script as follows.
# mpirun -n 2 [launcher-args] python ${base_dir}/../deepspeed_train.py
# As an example, below we include the command we used

base_dir=`pwd`

# Assumes job name in previous seq128 run, will resume training from epoch 150
EPOCH=150

# Where should we save checkpoints and tensorboard events?
JOB_NAME=onebit_lamb_32k_chkpt${EPOCH}_seq512_mpirun_infiniband
OUTPUT_DIR=${base_dir}/bert_model_outputs

CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/onebit_lamb_64k_seq128_mpirun_infiniband
CHECKPOINT_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch${EPOCH}_*`
echo "checkpoint id: $CHECKPOINT_NAME"

mkdir -p $OUTPUT_DIR

# LD_PRELOAD is used to load a specific nccl version. Remove it if needed.
mpirun -n 128 -ppn 8 -f /tmp/deepspeed_mvapich_hostfile -env MV2_SMP_USE_CMA=0 -env MV2_DEBUG_SHOW_BACKTRACE=1 -env MV2_USE_CUDA=1 -env MV2_SUPPORT_DL=1 -env MV2_ENABLE_AFFINITY=0 -env MV2_INTER_ALLGATHER_TUNING=5 -env MV2_CUDA_USE_NAIVE=0 -env NCCL_TREE_THRESHOLD=0 -env PYTHONUNBUFFERED=True -env MV2_USE_GDRCOPY=0 -env MV2_USE_GDR=0 -env NCCL_VERSION=2.8.3 -env PYTHON_VERSION=3 -env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3 python ${base_dir}/../deepspeed_train.py \
--cf ${base_dir}/../bert_large_lamb.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--print_steps 100 \
--deepspeed_mpi \
--deepspeed \
--deepspeed_transformer_kernel \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_bsz32k_onebitlamb_config_seq512_mpi_infiniband.json \
--data_path_prefix /data/bert \
--validation_data_path_prefix /data/bert \
--rewarmup \
--lr_schedule "EE" \
--attention_dropout_checkpoint \
--lr_offset 0.0 \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_NAME} \
&> ${JOB_NAME}.log
