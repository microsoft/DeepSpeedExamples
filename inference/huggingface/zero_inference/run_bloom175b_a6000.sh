#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference

# bloom 176b ds zero inference
MSZ="bloom"
QB=4
OFFLOAD_DIR=/local_nvme
mkdir -p $OFFLOAD_DIR

BSZ=8
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model bigscience/${MSZ} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --hf-model  --offload-dir ${OFFLOAD_DIR}  &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_hf_disk.txt 
deepspeed --num_gpus 1 run_model.py --dummy --model bigscience/${MSZ} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 1 --hf-model  --offload-dir ${OFFLOAD_DIR}  &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_hf_disk_pin.txt 
