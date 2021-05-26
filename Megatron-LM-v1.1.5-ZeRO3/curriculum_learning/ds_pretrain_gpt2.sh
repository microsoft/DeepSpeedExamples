#! /bin/bash

CONFIG=$1
TAG=$2
MODEL_SIZE=$3
LR=$4
TOTAL_BATCHSIZE=$5
SEQ_LEN=$6
NUM_ITER=$7
MP_SIZE=$8
LR_ITER=$9
LR_WARMUP=${10}
SEED=${11}

# 12-layer, 768-hidden, 12-heads, 117M parameters
# 24-layer, 1024-hidden, 16-heads, 345M parameters
# 36-layer, 1280-hidden, 20-heads, 774M parameters
# 48-layer, 1600-hidden, 25-heads, 1558M parameters
if [[ $MODEL_SIZE -eq 117 ]]; then
        NUM_LAYERS=12
        HIDDEN_SIZE=768
        NUM_ATTN_HEADS=12
elif [[ $MODEL_SIZE -eq 345 ]]; then
        NUM_LAYERS=24
        HIDDEN_SIZE=1024
        NUM_ATTN_HEADS=16
elif [[ $MODEL_SIZE -eq 774 ]]; then
        NUM_LAYERS=36
        HIDDEN_SIZE=1280
        NUM_ATTN_HEADS=20
elif [[ $MODEL_SIZE -eq 1558 ]]; then
        NUM_LAYERS=48
        HIDDEN_SIZE=1600
        NUM_ATTN_HEADS=25
else
        echo "Model size not supported."
        exit 1
fi

# Change for multinode config
NUM_WORKERS=4
NUM_GPUS_PER_WORKER=16
BATCHSIZE=$((MP_SIZE*TOTAL_BATCHSIZE/NUM_WORKERS/NUM_GPUS_PER_WORKER)) # per gpu batch size

# DATA_PATH=/data/Megatron-LM/data/indexed_datasets/megatron
# VOCAB_PATH=/data/Megatron-LM/data/gpt2-vocab.json
# MERGE_PATH=/data/Megatron-LM/data/gpt2-merges.txt

DATA_PATH=~/ssd/Megatron-LM/data/indexed_datasets/megatron
VOCAB_PATH=~/ssd/Megatron-LM/data/gpt2-vocab.json
MERGE_PATH=~/ssd/Megatron-LM/data/gpt2-merges.txt

#ZeRO Configs
stage=2
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
host="${HOSTNAME}"

config_json="$script_dir/ds_zero_stage_${stage}_config_${CONFIG}.json"
if [[ -z ${12} ]]; then
        JOB_NAME="gpt2_${MODEL_SIZE}M_lr${LR}_bsz${TOTAL_BATCHSIZE}_seql${SEQ_LEN}_iter${NUM_ITER}_lriter${LR_ITER}_warmup${LR_WARMUP}_seed${SEED}_${TAG}_stage${stage}_${NUM_WORKERS}n_${NUM_GPUS_PER_WORKER}g_${MP_SIZE}mp_${host}_${current_time}"
else
        JOB_NAME=${12}
fi

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

# Megatron Model Parallelism
LOGDIR="tboard/${JOB_NAME}"
CHECKPOINT_PATH="checkpoints/${JOB_NAME}"

gpt_options=" \
        --model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --batch-size $BATCHSIZE \
        --train-iters $NUM_ITER \
        --lr-decay-iters $LR_ITER \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr $LR \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup $LR_WARMUP \
        --checkpoint-activations \
        --log-interval 100 \
        --save-interval 1000 \
        --eval-interval 100 \
        --eval-iters 10 \
        --fp16 \
        --seed $SEED \
        --tensorboard-dir ${LOGDIR}
"

deepspeed_options=" \
        --deepspeed \
        --deepspeed_config ${config_json} \
        --zero-stage ${stage} \
        --zero-reduce-bucket-size ${rbs} \
        --zero-allgather-bucket-size ${agbs}
"

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --zero-reduce-scatter"
fi

chkp_opt=" \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi


full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  pretrain_gpt2.py ${@:13} ${full_options} &> ${JOB_NAME}.log"
echo ${run_cmd}
eval ${run_cmd}

set +x
