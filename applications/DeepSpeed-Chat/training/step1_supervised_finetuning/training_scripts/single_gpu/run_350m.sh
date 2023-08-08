#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Get the directory path of the current script file
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "The directory path of the current script file is: $CURRENT_DIR"

source $CURRENT_DIR/../../../base.sh

# Note that usually LoRA needs to use larger learning rate
OUTPUT=${1:-output}
ZERO_STAGE=${2:-0}
mkdir -p $OUTPUT

run_cmd="deepspeed --num_gpus 1 main.py \
  --model_name_or_path ${MODEL_STORAGE}facebook/opt-350m \
   --gradient_accumulation_steps 8 \
   --lora_dim 128 --zero_stage $ZERO_STAGE \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log"

echo ${run_cmd}
eval ${run_cmd}
