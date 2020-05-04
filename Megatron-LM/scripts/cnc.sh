#! /bin/bash

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
MP_SIZE=1


if [ -z $1 ];
then
    echo "Please set model size."
    exit 1
fi

MODEL_SIZE=$1

if [ -z $2 ];
then
    BUCKET_CAP_MB=25
else
    BUCKET_CAP_MB=$2
fi

if [ -z $3 ];
then
    TRAIN_ITERS=1000
else
    TRAIN_ITERS=$3
fi

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

if [[ ${MODEL_SIZE} == "320M" ]]; then
    #320M
    NUM_LAYERS=24
    HIDDEN_SIZE=1024
    NUM_ATT_HEADS=16
elif [[ ${MODEL_SIZE} == "1.2B" ]]; then
    #1.5B
    NUM_LAYERS=40
    HIDDEN_SIZE=1536
    NUM_ATT_HEADS=16
elif [[ ${MODEL_SIZE} == "1.5B" ]]; then
    #1.5B
    NUM_LAYERS=48
    HIDDEN_SIZE=1600
    NUM_ATT_HEADS=16
elif [[ ${MODEL_SIZE} == "3B" ]]; then
    #3B
    NUM_LAYERS=96
    HIDDEN_SIZE=1600
    NUM_ATT_HEADS=16
elif [[ ${MODEL_SIZE} == "5.9B" ]]; then
    #5.9B
    NUM_LAYERS=72
    HIDDEN_SIZE=2592
    NUM_ATT_HEADS=16
elif [[ ${MODEL_SIZE} == "8.3B" ]]; then
    # 8.3B
    NUM_LAYERS=72
    HIDDEN_SIZE=3072
    NUM_ATT_HEADS=32
    MP_SIZE=8
elif [[ ${MODEL_SIZE} == "10B" ]]; then
    # 10B
    NUM_LAYERS=85
    HIDDEN_SIZE=3104
    NUM_ATT_HEADS=32
elif [[ ${MODEL_SIZE} == "11B" ]]; then
    # 11B
    NUM_LAYERS=96
    HIDDEN_SIZE=3072
    NUM_ATT_HEADS=24
elif [[ ${MODEL_SIZE} == "13B" ]]; then
    # 13B
    NUM_LAYERS=114
    HIDDEN_SIZE=3072
    NUM_ATT_HEADS=24
elif [[ ${MODEL_SIZE} == "20B" ]]; then
    # 20B
    NUM_LAYERS=111
    HIDDEN_SIZE=3808
    NUM_ATT_HEADS=32
else
    echo "Unspecified model size ... exiting"
    exit -1
fi

gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATT_HEADS} \
       --batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters ${TRAIN_ITERS} \
       --resume-dataloader \
       --train-data webtext \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16 \
       --log-interval 1 \
       --bucket-cap-mb ${BUCKET_CAP_MB} \
"

       #--deepspeed \
       #--deepspeed_config ${config_json} \

kill_pdsh.sh
pkill -9 -f "nvidia-smi dmon"
[ ! -d "./log" ] && mkdir -p ./log
sleep 10

mem_cmd="nvidia-smi dmon -s m -i 1 &> log/cnc_mem_${MODEL_SIZE}_${BUCKET_CAP_MB}MB.log &"
run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} pretrain_gpt2.py ${gpt_options} &> log/cnc_${MODEL_SIZE}_${BUCKET_CAP_MB}MB.log"
echo ${run_cmd}
eval ${mem_cmd}
eval ${run_cmd}

set +x
