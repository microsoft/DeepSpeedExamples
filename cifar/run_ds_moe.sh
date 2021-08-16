#!/bin/bash

deepspeed cifar10_deepspeed.py \
	--log-interval 100 \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--moe \
	--ep-world-size 2 \
	--num-experts 2 \
	--top-k 1 \
	--noisy-gate-policy 'RSample' \
	--moe-param-group
