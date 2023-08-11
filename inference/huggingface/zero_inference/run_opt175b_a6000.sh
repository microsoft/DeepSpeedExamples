#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/

# 175b ds zeri inference
MSZ="opt-175b"
QB=4
OFFLOAD_DIR=/local_nvme/zero_offload
mkdir -p $OFFLOAD_DIR

BSZ=8
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --dummy --model facebook/${MSZ} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --hf-model  --offload-dir ${OFFLOAD_DIR} # &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_hf_disk.txt 

BSZ=32
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --dummy --model facebook/${MSZ} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --kv-offload --offload-dir ${OFFLOAD_DIR} # &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_disk.txt 

BSZ=24
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB}  &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_q${QB}.txt


# 175b flexgen with compute schedule or partial offloading
BSZ=64
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
OFFLOAD_DIR=/local_nvme/flexgen_offload
# mkdir -p $OFFLOAD_DIR
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1 # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_disk.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc_disk.txt  

# 175b flexgen with 4-bit weight quantization but without compute schedule or partial offloading
BSZ=40
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/opt-175b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_q4.txt    
# python -m flexgen.flex_opt --model facebook/opt-175b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute --compress-weight # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc_q4.txt    

