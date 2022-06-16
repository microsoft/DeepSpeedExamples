
export CUDA_VISIBLE_DEVICES=3
TASK_NAME=mnli  #mnli sst2 stsb mnli qqp rte cola mrpc qnli
STAGE=one_stage
#CONFIG=./config/ds_config_W1A8_64Qgroup_fp16.json # <=====================it's less stable
#CONFIG=./config/ds_config_W1or2A8_64Qgroup_fp16.json
CONFIG=./config/ds_config_W1A8_Qgroup1_fp32.json
SAVE_PATH=./new_result/old_code_output_G1_lr5e-5a/${TASK_NAME}_${STAGE}
mkdir -p ${SAVE_PATH}
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% if users provide *NO* models, use the following script %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% the following command will first download huggingface models and then compress %%%%%%%
MODEL=yoshitomo-matsubara/bert-base-uncased-${TASK_NAME} ## for both student and teacher
python -m torch.distributed.launch --nproc_per_node=1 \
  --master_port 66661 \
  run_glue.py \
  --seed 42 \
  --distill_method ${STAGE} \
  --model_name_or_path ${MODEL} \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --num_warmup_epochs 0 \
  --deepspeed_config ${CONFIG} --weight_bit 1 \
  --deepspeed \
  --save_best_checkpoint --clean_best_model \
  --gradient_accumulation_steps 1 \
  --output_dir ${SAVE_PATH} #&>> ${SAVE_PATH}/train.log