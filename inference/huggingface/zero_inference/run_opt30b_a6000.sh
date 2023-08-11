#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/

# 30b ds zeri inference
MSZ="opt-30b"

BSZ=24
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --hf-model # &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_hf_cpu_pin.txt 

BSZ=128
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload # &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu.txt 

BSZ=156
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
mkdir -p  $LOG_DIR
QB=4
deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload --quant_bit ${QB} &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_q${QB}.txt


# 30b flexgen with compute schedule or partial offloading
BSZ=200
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc.txt  

# 30b flexgen with 4-bit weight quantization but without compute schedule or partial offloading
BSZ=280
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_q4.txt    
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute --compress-weight &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc_q4.txt    

