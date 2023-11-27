NAME=test_1125a
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=1 \
train.py --cfg ./config/objaverse/train_1126_pflrm.yaml --name $NAME #&> out_${NAME}.log
# LEAP/config/,1,2,3,4,5,6,7,1,2,3,4,5,6,7