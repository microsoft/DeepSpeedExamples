NAME=largeXX_1124a
deepspeed --include worker-0 train_deepspeed.py --deepspeed_config ./ds_config.json \
--cfg ./config/objaverse/train_224_1122.yaml --name $NAME &> out_${NAME}.log
# LEAP/config/,1,2,3,4,5,6,7,1,2,3,4,5,6,7