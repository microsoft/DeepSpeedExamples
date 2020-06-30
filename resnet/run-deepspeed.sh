deepspeed \
	--num_nodes=$1 \
	--num_gpus=$2 \
	resnet_deepspeed.py \
	--train-dir=/data/ImageNet/train \
	--val-dir=/data/ImageNet/val/ \
	--deepspeed_config ds_config.json
