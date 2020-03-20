#! /bin/bash

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
MP_SIZE=2

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

if [ -z $1 ]; then
   config_json="$script_dir/ds_config.json"
else
   config_json="$script_dir/$1"
fi

NUM_NODES=${DLWS_NUM_WORKER}
NUM_GPUS_PER_NODE=${DLWS_NUM_GPU_PER_WORKER}

BATCH_SIZE=4

MODEL_SIZE="320M"

if [[ ${MODEL_SIZE} == "320M" ]]; then
   #320M
   NUM_LAYERS=24
   HIDDEN_SIZE=1024
   NUM_ATT_HEADS=16
elif [[ ${MODEL_SIZE} == "5.9B" ]]; then
   #5.9B
   NUM_LAYERS=72
   HIDDEN_SIZE=2592
   NUM_ATT_HEADS=32
elif [[ ${MODEL_SIZE} == "8.3B" ]]; then
   # 8.3B
   NUM_LAYERS=72
   HIDDEN_SIZE=3072
   NUM_ATT_HEADS=24
elif [[ ${MODEL_SIZE} == "10B" ]]; then
   # 10B
   NUM_LAYERS=85
   HIDDEN_SIZE=3104
   NUM_ATT_HEADS=32
elif [[ ${MODEL_SIZE} == "20B" ]]; then
   # 10B
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
       --batch-size ${BATCH_SIZE} \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100000 \
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
       --fp16
"
deepspeed_options=" \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --zero-stage 2 \
       --zero-reduce-bucket-size 600000000 \
       --zero-allgather-bucket-size 200000000 
"

checkpointing_options=" \
       --checkpoint-activations \
       --partition-activations \
       --checkpoint-in-cpu \
       --contigious-checkpointing \
       --synchronize-each-layer \
"
#       --profile-backward 

full_options="${gpt_options} ${deepspeed_options} ${checkpointing_options}"

run_cmd="deepspeed.pt --num_nodes ${NUM_NODES} --num_gpus ${NUM_GPUS_PER_NODE} pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
