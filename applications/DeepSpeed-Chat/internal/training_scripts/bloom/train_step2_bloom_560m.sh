#!/bin/bash
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company

# -----------------------------------------------------------------------
# RLHF step2 reference training script for Bloom-560m model
# -----------------------------------------------------------------------

set -ex

DATA_DIR_ROOT=${HL_DATA_DIR_ROOT:-/mnt/weka}
tag=${HL_TAG:-default_tag}
base_out_path=${HL_BASE_OUT_PATH:-/root/logs}
n_nodes=${HL_NUM_NODES:-1}
n_devices_per_node=${HL_DEVICES_PER_NODE:-8}
cri_zero_stage=${HL_CRITIC_ZERO_STAGE:-1}
seed=${HL_SEED:-10}
mbs=${HL_MBS:-8}
gbs=${HL_GBS:-64}
tensorboard_path=${HL_TENSORBOARD_PATH:-}
log_file=${HL_LOG_FILE:-}
checkpoint_path=${HL_CHECKPOINT_PATH:-}
master_port=${HL_MASTER_PORT:-29500}
model_name_or_path=${HL_CRITIC_MODEL:-bigscience/bloom-560m}
dataset_path=${HL_DATASET_PATH}
learning_rate=${HL_LEARNING_RATE:-2e-5}
lora_learning_rate=${HL_LORA_LEARNING_RATE:-2e-5}
weight_decay=${HL_WEIGHT_DECAY:-0.0}
epochs=${HL_EPOCHS:-2}
lora_dim=${HL_LORA_DIM:-0}
dropout=${HL_DROPOUT:-0.0}

# Calculate GAS given global batch, n_nodes, n_devices_per_node
total_devices=$(($n_nodes*$n_devices_per_node))
per_device_batch=$(($gbs/$total_devices))
gas=$(($per_device_batch/$mbs))

# setup checkpoint, tensorboard and log path
prefix_name=${tag}/bloom/step2/560m
run_name=gb_${gbs}_mbs_${mbs}_lr_${learning_rate}_do_${dropout}_wd_${weight_decay}_ep_${epochs}

lora_args=""
if [ "$lora_dim" -ne "0" ]; then
  lora_args="--lora_dim ${lora_dim} --lora_learning_rate ${lora_learning_rate} --lora_module_name rwtranrsformer.h. --only_optimize_lora "
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

CMD="step2_reward_model_finetuning/main.py \
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
        --num_padding_at_beginning 0 \
        --zero_stage ${cri_zero_stage} \
        --seed ${seed} \
        --optimized_reward_loss_calc \
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

#        --eval_interval 100 \
#        --eval_iters 100 \
