NAME=large_1125
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=16 \
train.py --cfg ./config/objaverse/train_224_1123b.yaml --name $NAME &> out_${NAME}.log
# LEAP/config/,1,2,3,4,5,6,7,1,2,3,4,5,6,7