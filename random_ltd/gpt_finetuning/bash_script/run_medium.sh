#!/bin/bash
##################apply random-ltd to fine-tune ptb on GPT-medium (24-layer)##############################
export CUDA_VISIBLE_DEVICES=1
mkdir -p ./output2/check-medium
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12346 \
    run_clm_no_trainer.py \
    --random_ltd \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path gpt2-medium \
    --per_device_train_batch_size 2 \
    --num_train_epochs 2 \
    --deepspeed_config config/ds_config_gpt_medium.json \
    --deepspeed --seed 1234\
    --output_dir ./output2/check-medium #&> ./output2/check/training.log 