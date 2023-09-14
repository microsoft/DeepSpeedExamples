#!/bin/bash
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company

# -----------------------------------------------------------------------
# RLHF step1 reference training script for Bloom-1.1B model
# -----------------------------------------------------------------------

set -ex

DATA_DIR_ROOT=${HL_DATA_DIR_ROOT:-/mnt/weka}
tag=${HL_TAG:-default_tag}
base_out_path=${HL_BASE_OUT_PATH:-/root/logs}
n_nodes=${HL_NUM_NODES:-1}
n_devices_per_node=${HL_DEVICES_PER_NODE:-8}
act_zero_stage=${HL_ACTOR_ZERO_STAGE:-1}
ckp_act=${HL_ACTOR_CP_ACT:-0}
seed=${HL_SEED:-10}
mbs=${HL_MBS:-8}
gbs=${HL_GBS:-128}
tensorboard_path=${HL_TENSORBOARD_PATH:-}
log_file=${HL_LOG_FILE:-}
checkpoint_path=${HL_CHECKPOINT_PATH:-}
master_port=${HL_MASTER_PORT:-29500}
model_name_or_path=${HL_ACTOR_MODEL:-bigscience/bloom-1b1}
dataset_path=${HL_DATASET_PATH}
learning_rate=${HL_LEARNING_RATE:-2e-5}
lora_learning_rate=${HL_LORA_LEARNING_RATE:-2e-5}
weight_decay=${HL_WEIGHT_DECAY:-0.0}
lora_dim=${HL_LORA_DIM:-0}
dropout=${HL_DROPOUT:-0.1}
epochs=${HL_EPOCHS:-4}

# Calculate GAS given global batch, n_nodes, n_devices_per_node
total_devices=$(($n_nodes*$n_devices_per_node))
per_device_batch=$(($gbs/$total_devices))
gas=$(($per_device_batch/$mbs))

# set gradient checkpointing arguments
ckp_act_args=""
if [ "$ckp_act" -eq "1" ]; then
  ckp_act_args="--gradient_checkpointing "
fi

# setup checkpoint, tensorboard and log path
prefix_name=${tag}/bloom/step1/1.1b
run_name=gb_${gbs}_mbs_${mbs}_lr_${learning_rate}_do_${dropout}_wd_${weight_decay}_ep_${epochs}

lora_args=""
if [ "$lora_dim" -ne "0" ]; then
  lora_args="--lora_dim ${lora_dim} --lora_learning_rate ${lora_learning_rate} --lora_module_name transformer.h. --only_optimize_lora "
  run_name=${run_name}_lora_lr_${lora_learning_rate}
fi

if [ -z "$tensorboard_path" ]; then
  tensorboard_path=${base_out_path}/tensorboard/${prefix_name}
fi

if [ -z "$log_file" ]; then
  log_file=${base_out_path}/logs/${prefix_name}/${run_name}.txt
fi

if [ -z "$checkpoint_path" ]; then
  checkpoint_path=${base_out_path}/checkpoints/${prefix_name}/${run_name}
fi

if [ "$n_nodes" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

# create required paths
# if log-file/tb-path provided, caller should make sure directories exist
mkdir -p ${base_out_path}/logs/${prefix_name}

# RUN
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
training_dir=$( realpath $script_dir/../../../training)
cd ${training_dir}

CMD="step1_supervised_finetuning/main.py \
        --model_name_or_path ${model_name_or_path} \
        --data_path ${dataset_path} \
        ${lora_args} \
        --dtype bf16 \
        --learning_rate ${learning_rate} \
        --dropout ${dropout} \
        --weight_decay ${weight_decay} \
        --per_device_train_batch_size ${mbs} \
        --gradient_accumulation_steps ${gas} \
        --num_train_epochs ${epochs} \
        --num_warmup_steps 20 \
        --zero_stage ${act_zero_stage} \
        ${ckp_act_args} \
        --per_device_eval_batch_size 8 \
        --seed ${seed} \
        --deepspeed \
        --output_dir ${checkpoint_path} \
        --enable_tensorboard \
        --tensorboard_path ${tensorboard_path} \
        --print_loss \
        --no_fused_kernels"

deepspeed --num_nodes ${n_nodes} \
          --num_gpus ${n_devices_per_node} \
          --master_port ${master_port} \
          $MULTINODE_CMD \
          $CMD   |& tee ${log_file}
exit $PIPESTATUS