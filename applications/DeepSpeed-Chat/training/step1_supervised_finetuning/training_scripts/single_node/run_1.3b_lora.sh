#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
# DeepSpeed Team
ZERO_STAGE=$1
OFFLOAD=$2
OUTPUT=$3
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
if [ "$OFFLOAD" == true ]; then
    OFFLOAD="--offload"
else
    OFFLOAD=""
fi
echo $OFFLOAD
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi

mkdir -p $OUTPUT

cmd="deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-1.3b \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0.1 \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --only_optimize_lora \
   --deepspeed \
   --output_dir $OUTPUT \
   $OFFLOAD"

echo $cmd

$cmd &> $OUTPUT/${OUTPUT}.log
