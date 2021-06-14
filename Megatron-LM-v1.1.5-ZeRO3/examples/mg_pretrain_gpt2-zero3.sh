#! /bin/bash

# Change for multinode config
MP_SIZE=1

DEBUG=1
if [[ ${DEBUG} == 1 ]];  then
       MP_SIZE=1
       NUM_WORKERS=1
       NUM_GPUS_PER_WORKER=1
       HIDDEN_SIZE=1600
       NUM_ATTN_HEADS=16
       NUM_LAYERS=40
       BATCHSIZE=8
else
       NUM_WORKERS=${DLTS_NUM_WORKER}
       NUM_GPUS_PER_WORKER=${DLTS_NUM_GPU_PER_WORKER}
       HIDDEN_SIZE=8192
       NUM_ATTN_HEADS=32
       NUM_LAYERS=50
       BATCHSIZE=4

       #HIDDEN_SIZE=4096
       #NUM_LAYERS=24 # 50
       #BATCHSIZE=16
fi


BASE_DATA_PATH=/data/Megatron-LM/data
DATA_PATH=${BASE_DATA_PATH}/indexed_datasets/megatron
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
if [[ -z $1 ]]; then
       config_json="$script_dir/ds_zero_stage_3_config.json"

       # offloads to NVMe
       #config_json="$script_dir/ds_zero_stage_infinity_config.json"
else
       config_json=$script_dir/`basename $1`
fi


# Megatron Model Parallelism
LOGDIR="tboard-zero3/stage${stage}-lazyscatter-${NUM_LAYERS}l_${HIDDEN_SIZE}h_${NUM_WORKERS}n_${NUM_GPUS_PER_WORKER}g_${MP_SIZE}mp_${BATCHSIZE}b"


gpt_options=" \
        --model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --lr 0.00015  \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --train-iters 100 \
        --lr-decay-iters 800 \
        --warmup 0.01 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 1000,0,0 \
        --fp16 \
        --checkpoint-activations \
        --log-interval 1 \
        --eval-iters 10 \
        --distributed-backend nccl \
"
full_options="${gpt_options}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
