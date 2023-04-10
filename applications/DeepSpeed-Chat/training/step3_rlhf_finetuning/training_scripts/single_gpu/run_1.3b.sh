#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


ACTOR_MODEL_PATH=/home/zheweiyao/ChatGPT_Pipeline/DeepSpeedExamples-internal/ChatGPT_RLHF_Example/step1_supervised_finetuning/final_coffee_break
CRITIC_MODEL_PATH=/home/zheweiyao/ChatGPT_Pipeline/DeepSpeedExamples-internal/ChatGPT_RLHF_Example/step2_reward_model_finetuning/coffee_break4_diferent_dataets



OUTPUT="./final_coffee_break_different_datasets"

mkdir -p $OUTPUT

deepspeed main.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 2 \
   --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing \
   --output_dir $OUTPUT &> $OUTPUT/training.log
