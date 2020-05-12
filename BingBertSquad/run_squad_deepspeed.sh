#~/bin/bash

#1: number of GPUs
#2: Model File Address
#3: BertSquad Data Directory Address
#4: Output Directory Address

NGPU_PER_NODE=$1
MODEL_FILE=$2
SQUAD_DIR=$3
OUTPUT_DIR=$4
LR=$5
SEED=$6
MASTER_PORT=$7
echo "arg6 is $6"
echo "arg7 is $7"

NUM_NODES=1
NGPU=$((NGPU_PER_NODE*NUM_NODES))
EFFECTIVE_BATCH_SIZE=24
MAX_GPU_BATCH_SIZE=6
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi
JOB_NAME="deepspeed_${NGPU}GPUs_${EFFECTIVE_BATCH_SIZE}batch_size"
config_json=deepspeed_bsz24_config.json
#run_cmd="deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NGPU_PER_NODE} \
#       --master_port=${MASTER_PORT} \
run_cmd="python3.6 -m torch.distributed.launch \
       --nproc_per_node=${NGPU} \
       --master_port=${MASTER_PORT} \
       nvidia_run_squad_deepspeed.py \
       --bert_model bert-large-uncased \
       --do_train \
       --do_lower_case \
       --predict_batch_size 3 \
       --do_predict \
       --train_file $SQUAD_DIR/train-v1.1.json \
       --predict_file $SQUAD_DIR/dev-v1.1.json \
       --train_batch_size $PER_GPU_BATCH_SIZE \
       --learning_rate ${LR} \
       --num_train_epochs 2.0 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir $OUTPUT_DIR \
       --job_name ${JOB_NAME} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --fp16 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --deepspeed_transformer_kernel \
       --model_file $MODEL_FILE \
       --seed ${SEED} \
       "
echo ${run_cmd}
eval ${run_cmd}
