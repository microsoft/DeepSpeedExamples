#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT_PATH=./release_test/output_1.3b_lora
mkdir -p $OUTPUT_PATH

deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-1.3b \
   --gradient_accumulation_steps 2 --lora_dim 128 \
   --deepspeed --output_dir $OUTPUT_PATH &> $OUTPUT_PATH/training.log
