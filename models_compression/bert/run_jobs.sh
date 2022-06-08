#!/bin/bash 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%. run_jobs mnli

export CUDA_VISIBLE_DEVICES=0
TASK_NAME=$1  #mnli sst2 stsb mnli qqp rte cola mrpc qnli
STAGE=one_stage
#CONFIG=./config/ds_config_W1A8_64Qgoup_fp16.json # <=====================it's less stable
CONFIG=./config/ds_config_W1A8_64Qgoup_fp32.json
SAVE_PATH=./output/${TASK_NAME}_${STAGE}
mkdir -p ${SAVE_PATH}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if user provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%the following command will first download huggingface models and then compress %%%%%%%
MODEL=yoshitomo-matsubara/bert-base-uncased-${TASK_NAME} ## for both student and teacher
python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 66667 \
  run_glue.py \
  --seed 42 \
  --distill_method ${STAGE} \
  --model_name_or_path ${MODEL} \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 18 \
  --deepspeed_config ${CONFIG} --weight_bit 1 \
  --deepspeed \
  --save_best_checkpoint --save_last_model --clean_last_model \
  --gradient_accumulation_steps 1 \
  --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% users provide models  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODEL_BASE=/blob/users/xwu/compression/huggingface_models/bert_base_uncased ## or you could use bert-base-uncased 
# TEACHER=/blob/users/xwu/compression/huggingface_models/bert-base-uncased-${TASK_NAME}/pytorch_model.bin 
# STUDENT=${TEACHER}
# python -m torch.distributed.launch --nproc_per_node=1 \
#   --master_port 66667 \
#   run_glue.py \
#   --seed 42 \
#   --distill_method ${STAGE} \
#   --model_name_or_path ${MODEL_BASE} \
#   --pretrained_dir_student ${STUDENT} \
#   --pretrained_dir_teacher ${TEACHER} \
#   --task_name $TASK_NAME \
#   --max_length 128 \
#   --pad_to_max_length \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 18 \
#   --deepspeed_config ${CONFIG} --weight_bit 1 \
#   --deepspeed \
#   --save_best_checkpoint --save_last_model --clean_last_model \
#   --gradient_accumulation_steps 1 \
#   --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log 