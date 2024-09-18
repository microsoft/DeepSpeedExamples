#!/bin/bash

export PYTHONPATH=/workspace/domino/Megatron-LM/$PYTHONPATH
export LD_LIBRARY_PATH=/workspace/apex:$LD_LIBRARY_PATH 
export CUDA_DEVICE_MAX_CONNECTIONS=1
 
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
 
CHECKPOINT_PATH=/workspace/dataset/checkpoint
rm -rf $CHECKPOINT_PATH/*
rm -rf ./wandb/*
VOCAB_FILE="/workspace/dataset/gpt2-vocab.json"
MERGE_FILE="/workspace/dataset/gpt2-merges.txt"
DATA_PATH="/workspace/dataset/BookCorpusDataset_text_document"
 
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
 
# LLaMA2 7b
#    --no-async-tensor-model-parallel-allreduce \
#     --disable-bias-linear \
# LLAMA_ARGS="
#     --num-layers 32 \
#     --hidden-size 4096 \
#     --num-attention-heads 32 \
#     --seq-length 1024 \
#     --max-position-embeddings 1024 \
#     --position-embedding-type rope \
#     --swiglu \
#     --ffn-hidden-size 11008\
#     --disable-bias-linear \
#     --normalization RMSNorm \
#     --layernorm-epsilon 1e-6 \
#     --micro-batch-size 16 \
#     --global-batch-size 16 \
#     --lr 0.00015 \
#     --train-iters 80 \
#     --lr-decay-iters 320000 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --no-gradient-accumulation-fusion \
#     --fp16 \
#     --tensor-model-parallel-size $WORLD_SIZE \
#     --seed 3407 \
#     --causal-lm
# "

# llama2-13b
LLAMA_ARGS="
    --llama-model \
    --num-layers 2 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --position-embedding-type rope \
    --swiglu \
    --ffn-hidden-size 11008 \
    --disable-bias-linear \
    --normalization RMSNorm \
    --layernorm-epsilon 1e-6 \
    --micro-batch-size 8 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 100 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion \
    --fp16 \
    --tensor-model-parallel-size $WORLD_SIZE \
    --seed 3407 \
    --causal-lm
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"
 
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1
"
 
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
    $LLAMA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl #\
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
 
