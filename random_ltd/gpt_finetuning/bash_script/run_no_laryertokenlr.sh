#!/bin/bash
##################fine-tune the origin model and then apply zeroquant, the following command will take approximately 10 mins in A100
###zero-quant https://arxiv.org/abs/2206.01861
export CUDA_VISIBLE_DEVICES=1

######### fp16
# python -m torch.distributed.launch --nproc_per_node=1 \
#     --master_port 12345 \
#     run_clm_no_trainer.py \
#     --dataset_name ptb_text_only \
#     --dataset_config_name penn_treebank \
#     --model_name_or_path gpt2 \
#     --per_device_train_batch_size 4 \
#     --num_train_epochs 2 \
#     --deepspeed_config config/ds_config_old.json \
#     --deepspeed --seed 1234\
#     --output_dir ./output/check | tee -a ./output/check/train_before2.log

# export CUDA_VISIBLE_DEVICES=0
# mkdir output1/check
# ######### fp16
python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port 12346 \
    run_clm_no_trainer.py \
    --dataset_name ptb_text_only \
    --dataset_config_name penn_treebank \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 2 \
    --deepspeed_config config/ds_config.json \
    --deepspeed --seed 1234\
    --output_dir ./output1/check &> ./output1/check/train_with_real_kernel.log # ./output1/check/train_with_kernel_fp16.log
