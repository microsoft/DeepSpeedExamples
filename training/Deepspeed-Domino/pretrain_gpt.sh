	# #!/bin/bash --login
	# export PYTHONPATH=/workspace/domino/Megatron-LM:$PYTHONPATH
	 
	# # Runs the "345M" parameter model
	 
	# export CUDA_DEVICE_MAX_CONNECTIONS=1
	 
	# GPUS_PER_NODE=4
	# # Change for multinode config
	# MASTER_ADDR=localhost
	# # MASTER_ADDR=deepspeed-000000
	# MASTER_PORT=6001
	# NNODES=1
	# NODE_RANK=0
	# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
	 
	# CHECKPOINT_PATH=/workspace/dataset/checkpoint
	# rm -rf $CHECKPOINT_PATH/*
	# rm -rf ./wandb/*
	# VOCAB_FILE="/workspace/dataset/gpt2-vocab.json"
	# MERGE_FILE="/workspace/dataset/gpt2-merges.txt"
	# DATA_PATH="/workspace/dataset/BookCorpusDataset_text_document"
	
	# DISTRIBUTED_ARGS="
	#     --nproc_per_node $GPUS_PER_NODE \
	#     --nnodes $NNODES \
	#     --node_rank $NODE_RANK \
	#     --master_addr $MASTER_ADDR \
	#     --master_port $MASTER_PORT
	# "
	# # GPT-3 2.7B
	# # --no-async-tensor-model-parallel-allreduce \
	# # GPT_ARGS="
	# #     --num-layers 32 \
	# #     --hidden-size 2560 \
	# #     --num-attention-heads 32 \
	# #     --seq-length 512 \
	# #     --max-position-embeddings 512 \
	# #     --micro-batch-size 64 \
	# #     --global-batch-size 64 \
	# #     --lr 0.00015 \
	# #     --train-iters 80 \
	# #     --lr-decay-iters 320000 \
	# #     --lr-decay-style cosine \
	# #     --min-lr 1.0e-5 \
	# #     --weight-decay 1e-2 \
	# #     --lr-warmup-fraction .01 \
	# #     --clip-grad 1.0 \
	# #     --no-gradient-accumulation-fusion \
	# #     --fp16 \
	# #     --tensor-model-parallel-size $WORLD_SIZE \
	# #     --seed 3407
	# # "
	# #    --use-flash-attn
	# ## GPT-3 6.7B
	# # GPT_ARGS="
	# #     --num-layers 32 \
	# #     --hidden-size 4096 \
	# #     --num-attention-heads 32 \
	# #     --seq-length 2048 \
	# #     --max-position-embeddings 2048 \
	# #     --micro-batch-size 8 \
	# #     --global-batch-size 8 \
	# #     --lr 0.00015 \
	# #     --train-iters 80 \
	# #     --lr-decay-iters 320000 \
	# #     --lr-decay-style cosine \
	# #     --min-lr 1.0e-5 \
	# #     --weight-decay 1e-2 \
	# #     --lr-warmup-fraction .01 \
	# #     --clip-grad 1.0 \
	# #     --no-gradient-accumulation-fusion \
	# #     --fp16 \
	# #     --tensor-model-parallel-size $WORLD_SIZE
	# # "
	
	# #--num-attention-heads 40 \
	# # mb 16 oom even act-ckpt
	# # mb 32 oom for 4 nodes
	# # 13B
	# #    --use-flash-attn
	# # --no-async-tensor-model-parallel-allreduce \
	# # self.dense overlap compute comm
	# #    --disable-bias-linear
	# GPT_ARGS="
	#     --num-layers 2 \
	#     --hidden-size 4096 \
	#     --num-attention-heads 32 \
	#     --seq-length 1024 \
	#     --max-position-embeddings 1024 \
	#     --micro-batch-size 8 \
	#     --global-batch-size 8 \
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
	#     --tensor-model-parallel-size $WORLD_SIZE
	# "
	
	# # 30B
	# # 30B 4nodes mb16 OOM
	# # --recompute-activations
	# # GPT_ARGS="
	# #     --num-layers 64 \
	# #     --hidden-size 6144 \
	# #     --num-attention-heads 64 \
	# #     --seq-length 512 \
	# #     --max-position-embeddings 512 \
	# #     --micro-batch-size 4 \
	# #     --global-batch-size 4 \
	# #     --lr 0.00015 \
	# #     --train-iters 80 \
	# #     --lr-decay-iters 320000 \
	# #     --lr-decay-style cosine \
	# #     --min-lr 1.0e-5 \
	# #     --weight-decay 1e-2 \
	# #     --lr-warmup-fraction .01 \
	# #     --clip-grad 1.0 \
	# #     --no-gradient-accumulation-fusion \
	# #     --fp16 \
	# #     --tensor-model-parallel-size $WORLD_SIZE \
	# #     --use-flash-attn
	# # "
	
	
	# # GPT_ARGS="
	# #     --tensor-model-parallel-size $WORLD_SIZE \
	# #     --num-layers 24 \
	# #     --hidden-size 1024 \
	# #     --num-attention-heads 16 \
	# #     --seq-length 1024 \
	# #     --max-position-embeddings 1024 \
	# #     --micro-batch-size 8 \
	# #     --global-batch-size 8 \
	# #     --lr 0.00015 \
	# #     --train-iters 20 \
	# #     --lr-decay-iters 3200 \
	# #     --lr-decay-style cosine \
	# #     --min-lr 1.0e-5 \
	# #     --weight-decay 1e-2 \
	# #     --lr-warmup-fraction .01 \
	# #     --clip-grad 1.0 \
	# #     --fp16 \
	# #     --loss-scale 128 \
	# #     --seed 3407
	# # "
	
	# DATA_ARGS="
	#     --data-path $DATA_PATH \
	#     --vocab-file $VOCAB_FILE \
	#     --merge-file $MERGE_FILE \
	#     --split 949,50,1
	# "
	 
	# OUTPUT_ARGS="
	#     --log-interval 1 \
	#     --save-interval 10000 \
	#     --eval-interval 1000 \
	#     --eval-iters 1
	# "
	# # export NCCL_DEBUG=INFO
	# # export NCCL_NTHREADS=1
	# # export NCCL_MAX_NCHANNELS=1
	# # export NCCL_MIN_NCHANNELS=1
	# # export NCCL_BUFFSIZE=1048576
	# # export NCCL_SOCKET_NTHREADS=4
	# # export NCCL_NSOCKS_PERTHREAD=8
	
	# # cd /work/guanhua/domino
	# # python3 -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
	# # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
	# CUDA_VISIBLE_DEVISES=0,1,2,3 torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
	#     $GPT_ARGS \
	#     $DATA_ARGS \
	#     $OUTPUT_ARGS \
	#     --distributed-backend nccl #\
	#     # --save $CHECKPOINT_PATH \
    # # --load $CHECKPOINT_PATH


	#!/bin/bash --login
	export PYTHONPATH=/home/guanhua/zcm/domino/Megatron-LM:$PYTHONPATH
	
	# Runs the "345M" parameter model
	 
	export CUDA_DEVICE_MAX_CONNECTIONS=1
	 
	GPUS_PER_NODE=4
	# Change for multinode config
	MASTER_ADDR=localhost
	# MASTER_ADDR=deepspeed-000000
	MASTER_PORT=6001
	NNODES=1
	NODE_RANK=0
	WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
	 
	CHECKPOINT_PATH=/home/guanhua/zcm/checkpoint
	rm -rf $CHECKPOINT_PATH/*
	rm -rf ./wandb/*
	VOCAB_FILE="/home/guanhua/dataset/gpt2-vocab.json"
	MERGE_FILE="/home/guanhua/dataset/gpt2-merges.txt"
	DATA_PATH="/home/guanhua/dataset/BookCorpusDataset_text_document"
	
	DISTRIBUTED_ARGS="
	    --nproc_per_node $GPUS_PER_NODE \
	    --nnodes $NNODES \
	    --node_rank $NODE_RANK \
	    --master_addr $MASTER_ADDR \
	    --master_port $MASTER_PORT
	"
	# GPT-3 2.7B
	# --no-async-tensor-model-parallel-allreduce \
	# GPT_ARGS="
	#     --num-layers 32 \
	#     --hidden-size 2560 \
	#     --num-attention-heads 32 \
	#     --seq-length 512 \
	#     --max-position-embeddings 512 \
	#     --micro-batch-size 64 \
	#     --global-batch-size 64 \
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
	#     --seed 3407
	# "
	#    --use-flash-attn
	## GPT-3 6.7B
	# GPT_ARGS="
	#     --num-layers 32 \
	#     --hidden-size 4096 \
	#     --num-attention-heads 32 \
	#     --seq-length 2048 \
	#     --max-position-embeddings 2048 \
	#     --micro-batch-size 8 \
	#     --global-batch-size 8 \
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
	#     --tensor-model-parallel-size $WORLD_SIZE
	# "
	
	#--num-attention-heads 40 \
	# mb 16 oom even act-ckpt
	# mb 32 oom for 4 nodes
	# 13B
	#    --use-flash-attn
	# --no-async-tensor-model-parallel-allreduce \
	# self.dense overlap compute comm
	#    --disable-bias-linear
	GPT_ARGS="
	    --num-layers 2 \
	    --hidden-size 4096 \
	    --num-attention-heads 32 \
	    --seq-length 1024 \
	    --max-position-embeddings 1024 \
	    --micro-batch-size 8 \
	    --global-batch-size 8 \
	    --lr 0.00015 \
	    --train-iters 10 \
	    --lr-decay-iters 320000 \
	    --lr-decay-style cosine \
	    --min-lr 1.0e-5 \
	    --weight-decay 1e-2 \
	    --lr-warmup-fraction .01 \
	    --clip-grad 1.0 \
	    --fp16 \
	    --tensor-model-parallel-size $WORLD_SIZE
	"
	
	# 30B
	# 30B 4nodes mb16 OOM
	# --recompute-activations
	# GPT_ARGS="
	#     --num-layers 64 \
	#     --hidden-size 6144 \
	#     --num-attention-heads 64 \
	#     --seq-length 512 \
	#     --max-position-embeddings 512 \
	#     --micro-batch-size 4 \
	#     --global-batch-size 4 \
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
	#     --use-flash-attn
	# "
	
	
	# GPT_ARGS="
	#     --tensor-model-parallel-size $WORLD_SIZE \
	#     --num-layers 24 \
	#     --hidden-size 1024 \
	#     --num-attention-heads 16 \
	#     --seq-length 1024 \
	#     --max-position-embeddings 1024 \
	#     --micro-batch-size 8 \
	#     --global-batch-size 8 \
	#     --lr 0.00015 \
	#     --train-iters 20 \
	#     --lr-decay-iters 3200 \
	#     --lr-decay-style cosine \
	#     --min-lr 1.0e-5 \
	#     --weight-decay 1e-2 \
	#     --lr-warmup-fraction .01 \
	#     --clip-grad 1.0 \
	#     --fp16 \
	#     --loss-scale 128 \
	#     --seed 3407
	# "
	
	DATA_ARGS="
	    --data-path $DATA_PATH \
	    --vocab-file $VOCAB_FILE \
	    --merge-file $MERGE_FILE \
	    --split 949,50,1 \
        --tokenizer-type GPT2BPETokenizer
	"
	 
	# OUTPUT_ARGS="
	#     --log-interval 1 \
	#     --save-interval 10000 \
	#     --eval-interval 1000 \
	#     --eval-iters 1
	# "
    OUTPUT_ARGS = " "

	# export NCCL_DEBUG=INFO
	# export NCCL_NTHREADS=1
	# export NCCL_MAX_NCHANNELS=1
	# export NCCL_MIN_NCHANNELS=1
	# export NCCL_BUFFSIZE=1048576
	# export NCCL_SOCKET_NTHREADS=4
	# export NCCL_NSOCKS_PERTHREAD=8
	
	# cd /work/guanhua/domino
	# python3 -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
	# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
	# CUDA_VISIBLE_DEVISES=0,1,2,3 torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
	python3 -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
	    $GPT_ARGS \
	    $DATA_ARGS \
	    $OUTPUT_ARGS \
	    --distributed-backend nccl #\
	    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH