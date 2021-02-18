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

mpirun -n 128 -ppn 8 -f /tmp/deepspeed_mvapich_hostfile -env MV2_SUPPORT_DL=1 python ${base_dir}/../deepspeed_train.py \
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
