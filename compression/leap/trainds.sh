NAME=largeXX_1124a
deepspeed train_deepspeed.py --cfg ./config/objaverse/train_224_1122.yaml --name $NAME &> out_${NAME}.log
# LEAP/config/,1,2,3,4,5,6,7,1,2,3,4,5,6,7