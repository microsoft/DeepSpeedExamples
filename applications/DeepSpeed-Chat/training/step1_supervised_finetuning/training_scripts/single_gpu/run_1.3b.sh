#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE="auto"
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 2 main.py \
   --model_name_or_path facebook/opt-1.3b \
   --gradient_accumulation_steps 2 \
   --lora_dim 128 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --per_device_train_batch_size 16 \
   --output_dir $OUTPUT #&> $OUTPUT/training.log
