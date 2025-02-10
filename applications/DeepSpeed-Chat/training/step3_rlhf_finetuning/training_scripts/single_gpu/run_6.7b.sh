#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=${1:-"../output/actor_model"}
CRITIC_MODEL_PATH=${2:-"../output/critic_model"}
ACTOR_ZERO_STAGE=${3:-"--actor_zero_stage 0"}
CRITIC_ZERO_STAGE=${4:-"--critic_zero_stage 0"}
OUTPUT=${5:-"./output"}


Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=5e-4
Critic_Lr=5e-6

mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 8 \
   --per_device_mini_train_batch_size 8 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --ppo_epochs 1 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 16 \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} ${OFFLOAD}\
   --actor_lora_dim 128 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
