OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

if [[ $0 =~ ^\/.* ]]   
then
  script=$0
else
  script=$(pwd)/$0
fi
path_dir=${script%%training_scripts*}
echo $path_dir

ds --num_gpus 2 $path_dir'main.py' \
   --data_path $HOME/.cache/huggingface/hub/datasets--Dahoas--full-hh-rlhf \
   --data_split 2,4,4 \
   --model_name_or_path bigscience/bloom-560m \
   --tokenizer_name_or_path bigscience/tokenizer \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --disable_dropout \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   > $OUTPUT/training_step2_bloom_560m_dahoas_full_hh_rlhf.log 2>&1 &
