~/.local/bin/horovodrun -np $1  \
	-hostfile ~/hosts \
	--fusion-threshold-mb $2 \
	python \
	resnet_horovod.py \
	--train-dir=/data/ImageNet/train \
	--val-dir=/data/ImageNet/val \
	--batch-size=128


