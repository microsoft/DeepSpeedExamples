#!/bin/bash

#"train_micro_batch_size_per_gpu" : 16,

#deepspeed -i worker-0@worker-1 train.py --deepspeed_config=ds_config.json -p 2 2>&1 --steps=20000 | tee ${LOGFILE}
GPUS=4
for PP in 2 ;
do
    if [ $PP -gt 0 ];
    then
        DP=$(( ${GPUS} / ${PP} ))
    else
        DP=${GPUS}
    fi

    LOGFILE="results/cifar10-batch256-gpus${GPUS}-dp${DP}-pp${PP}-mb32.log"
    deepspeed --num_nodes=1 train.py --deepspeed_config=ds_config.json -p ${PP} 2>&1 --steps=200 | tee ${LOGFILE}
done

deepspeed ~/train.py