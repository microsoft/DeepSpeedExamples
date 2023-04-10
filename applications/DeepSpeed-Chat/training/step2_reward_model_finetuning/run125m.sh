#!/bin/bash
OUTPUT_PATH=$1
ZERO_STAGE=$2
mkdir -p $OUTPUT_PATH

CUDA_VISIBLE_DEVICES=1 deepspeed --num_gpus 1 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-125m \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 64 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT_PATH #\
   #&> $OUTPUT_PATH/training.log
   
#Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP \
