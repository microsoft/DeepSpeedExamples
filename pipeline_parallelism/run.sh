#!/bin/bash

PP=2
LOGFILE="results/cifar10-batch256-dp4-pp2-mb4-part_parameters.log"

deepspeed -i worker-0@worker-1 train.py --deepspeed_config=ds_config.json -p 2 2>&1 --steps=20000 | tee ${LOGFILE}

deepspeed ~/train.py