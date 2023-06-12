#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/bloomz-560m.phoenix_v1_test5
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
   --data_path custom/phoenix_v1 \
   --model_name_or_path bigscience/bloomz-560m \
   --data_split 2,4,4 --per_device_train_batch_size 2 --per_device_eval_batch_size 2  --lr_scheduler_type cosine --num_warmup_steps 0 \
   --gradient_accumulation_steps 1 --zero_stage $ZERO_STAGE --local_rank 0 \
   --max_seq_len 512 --learning_rate 9.65e-6 --weight_decay 0. --num_train_epochs 1 \
   --deepspeed --output_dir $OUTPUT 2>&1 | tee $OUTPUT/training.log \
   