#!/bin/bash

while [ 1 ]; do
    latest=`ls -t *_*.log | head -1`
    echo $latest
    date
    grep "CUDA out of memory" $latest
    if [ $? == 0 ]; then
        echo "Found max deepscale model size, done."
        ds_ssh pkill python
        sleep 900
        continue
    fi
    grep "NCCL error" $latest
    if [ $? == 0 ]; then
        echo "Found nccl error with model"
        ds_ssh pkill python
        sleep 900
        continue
    fi
    grep "was not found in model name list (gpt2)" $latest
    if [ $? == 0 ]; then
        echo "Found gpt download error with model"
        ds_ssh pkill python
        sleep 900
        continue
    fi

    grep "CUDA error: an illegal memory access" $latest
    if [ $? == 0 ]; then
        echo "Found memory error model"
        ds_ssh pkill python
        sleep 900
        continue
    fi   

    sleep 900
done
