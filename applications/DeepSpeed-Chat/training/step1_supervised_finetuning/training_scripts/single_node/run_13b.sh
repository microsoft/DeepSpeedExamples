#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT_PATH=./release_test/output_13b
mkdir -p $OUTPUT_PATH

deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-13b \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   &> $OUTPUT_PATH/training.log
