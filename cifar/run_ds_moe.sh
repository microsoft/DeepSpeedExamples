#!/bin/bash

deepspeed cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size 2 \
	--num-experts 2 \
	--top-k 2 \
	--moe-param-group
