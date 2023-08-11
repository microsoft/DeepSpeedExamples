#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/

# 66b ds zeri inference
MSZ="opt-66b"
QB=4 


BSZ=16
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --hf-model #&> $LOG_DIR/ds_${MSZ}_bs${BSZ}_hf_cpu_pin.txt 

BSZ=40
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload #&> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu.txt 


BSZ=64
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload --quant_bit ${QB} &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_q${QB}.txt


# 66b flexgen with compute schedule or partial offloading
MSZ="opt-66b"
BSZ=80
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 #&> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute #&> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc.txt  


# 66b flexgen with 4-bit weight quantization but without compute schedule or partial offloading
BSZ=96
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_q4.txt    

BSZ=100
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute --compress-weight # &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc_q4.txt    

