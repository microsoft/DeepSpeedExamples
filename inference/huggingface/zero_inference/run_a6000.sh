#!/bin/sh
export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
# 1.3b ds zeri inference
MSZ="opt-1.3b"
BSZ=200
QB=4
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu.txt 
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --kv-offload &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_pin.txt
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --kv-offload --quant_bit 4 &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_pin_q${QB}.txt
# 1.3b flexgen with compute schedule or partial offload
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 1 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_pin.txt

# mkdir -p  $LOG_DIR
# OFFLOAD_DIR=/local_nvme/flexgen_offload
# mkdir -p $OFFLOAD_DIR
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_disk.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc_disk.txt  


# 30b ds zeri inference
MSZ="opt-30b"

BSZ=24
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --hf-model &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_hf_cpu.txt 
# deepspeed --num_gpus 1 run_model.py --model facebook/${MSZ} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --hf-model &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_hf_cpu_pin.txt 

BSZ=128
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu.txt 
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload --quant_bit 4 &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_q${QB}.txt
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --kv-offload --quant_bit 4 &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_pin_q${QB}.txt

BSZ=156
LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload --quant_bit 4 &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_q${QB}.txt
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 1 --kv-offload --quant_bit 4 &> $LOG_DIR/ds_${MSZ}_bs${BSZ}_cpu_pin_q${QB}.txt

# 30b ds zeri inference
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size 48 --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size 128 --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size 128 --cpu-offload --gen-len 32 --pin-memory 0 --kv-offload --quant_bit 4


# 30b ds zeri inference with 4-bit weight qunatization
# deepspeed --num_gpus 1 run_model.py --model facebook/opt-30b --batch-size 156 --cpu-offload --gen-len 32 --kv-offload --pin-memory 0 --quant_bit 4


# 30 flexgen with compute schedule or partial offloading
# python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 1 --num-gpu-batches 1
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 1
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 200 --num-gpu-batches 1
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 200 --num-gpu-batches 1 --cpu-cache-compute

# 30 flexgen with 4-bit weight quantization but without compute schedule or partial offloading
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 280 --num-gpu-batches 1 --compress-weight
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 280 --num-gpu-batches 1 --cpu-cache-compute --compress-weight


# 30 flexgen with compute schedule or partial offloading
# MSZ="opt-30b"
# BSZ=200
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 1 --num-gpu-batches 1 &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_pin.txt
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc.txt  
# python -m flexgen.flex_opt --model facebook/${MSZ} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 1 --num-gpu-batches 1 --cpu-cache-compute &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_pin_ccc.txt 

# # 30 flexgen with 4-bit weight quantization but without compute schedule or partial offloading
# BSZ=280
# LOG_DIR=$BASE_LOG_DIR/${MSZ}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 280 --pin-weight 0 --num-gpu-batches 1 --compress-weight &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_q4.txt    
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 280 --pin-weight 1 --num-gpu-batches 1 --compress-weight &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_pin_q4.txt 
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 280 --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute --compress-weight &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_ccc_q4.txt    
# python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 280 --pin-weight 1 --num-gpu-batches 1 --cpu-cache-compute --compress-weight &> $LOG_DIR/fg_${MSZ}_bs${BSZ}_cpu_pin_ccc_q4.txt 

