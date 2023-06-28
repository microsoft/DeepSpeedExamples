#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
ENABLE_HYBRID_ENGINE=$5
OFFLOAD=$6
UNPIN_ACTOR_PARAMETERS=$7
BASE=$8
OUTPUT=$9
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi

if [ "$ENABLE_HYBRID_ENGINE" == true ]; then
    ENABLE_HYBRID_ENGINE="--enable_hybrid_engine"
else
    ENABLE_HYBRID_ENGINE=""
fi

if [ "$OFFLOAD" == true ]; then
    OFFLOAD="--offload"
else
    OFFLOAD=""
fi

if [ "$UNPIN_ACTOR_PARAMETERS" == true ]; then
    UNPIN_ACTOR_PARAMETERS="--unpin_actor_parameters"
else
    UNPIN_ACTOR_PARAMETERS=""
fi

if [ "$BASE" == '' ]; then
	BASE="True"
fi

export BASE=${BASE}
echo "BASE=" $BASE

echo $ENABLE_HYBRID_ENGINE
echo $OFFLOAD
echo $UNPIN_ACTOR_PARAMETERS

mkdir -p $OUTPUT

Num_Padding_at_Beginning=1 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

#cmd="deepspeed --num_gpus 2 --master_port 12346 main.py \
cmd="deepspeed --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 2 \
   --per_device_mini_train_batch_size 2 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_ema \
   --output_dir $OUTPUT \
    $ENABLE_HYBRID_ENGINE $OFFLOAD $UNPIN_ACTOR_PARAMETERS"

echo $cmd

$cmd &> $OUTPUT/${OUTPUT}.log
#$cmd
