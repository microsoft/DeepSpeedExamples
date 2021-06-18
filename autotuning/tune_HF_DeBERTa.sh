batch_size=4
TASK_NAME=mrpc
output_dir=/tmp/mrpc_out
HF_PATH=/home/minjiaz/workspace/tuning/

deepspeed --autotuning tune $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed_config ds_config.json \
  --model_name_or_path microsoft/deberta-v2-xxlarge \
  --task_name ${TASK_NAME} \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 13e-6 \
  --num_train_epochs 3\
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --logging_dir ${output_dir} \
  --save_steps 0