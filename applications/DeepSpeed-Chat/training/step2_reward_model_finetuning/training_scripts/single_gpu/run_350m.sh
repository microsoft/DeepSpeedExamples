#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT_PATH=./coffee_break4_diferent_dataets
mkdir -p $OUTPUT_PATH

deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-350m \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 2 \
   --deepspeed --output_dir $OUTPUT_PATH &> $OUTPUT_PATH/training.log
