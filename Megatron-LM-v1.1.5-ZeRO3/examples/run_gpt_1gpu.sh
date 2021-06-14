NUM_WORKERS=1
NUM_GPUS_PER_WORKER=16
MP_SIZE=1
NUM_LAYERS=40
HIDDEN_SIZE=1600
NUM_ATTN_HEADS=16
BATCHSIZE=8

BASE_DATA_PATH=/data/Megatron-LM/data
DATA_PATH=${BASE_DATA_PATH}/indexed_datasets/megatron
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds

gpt_options="
    --model-parallel-size ${MP_SIZE} \
    --num-layers ${NUM_LAYERS}   \
    --hidden-size $HIDDEN_SIZE  \
    --num-attention-heads ${NUM_ATTN_HEADS}  \
    --seq-length 1024  \
    --max-position-embeddings 1024  \
    --batch-size $BATCHSIZE \
    --lr 0.00015  \
    --lr-decay-style cosine  \
    --min-lr 1.0e-5  \
    --train-iters 5  \
    --lr-decay-iters 800   \
    --warmup 0.01 \
    --weight-decay 1e-2  \
    --clip-grad 1.0 \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_PATH \
    --merge-file $MERGE_PATH \
    --data-impl mmap  \
    --split 1000,0,0 \
    --fp16  \
    --checkpoint-activations \
    --log-interval 1  \
    --eval-iters 10  \
    --distributed-backend nccl \
"

full_options="${gpt_options}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  pretrain_gpt2.py ${@:2} ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
