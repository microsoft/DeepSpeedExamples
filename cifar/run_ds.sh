#!/bin/bash

num_gpus=1
ds_config=ds_config.json

args="--deepspeed --deepspeed_config ${ds_config}"
run_cmd="deepspeed.pt --num_nodes 1 --num_gpus ${num_gpus} cifar10_deepspeed.py $@ ${args}"
echo ${run_cmd}
eval ${run_cmd}
