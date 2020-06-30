deepspeed \
	--num_nodes=$2 \
	--num_gpus=$1 \
	resnet_deepspeed.py \
	--train-dir=/data/ImageNet/train \
	--val-dir=/data/ImageNet/val/ \
	--deepspeed_config ds_config.json
