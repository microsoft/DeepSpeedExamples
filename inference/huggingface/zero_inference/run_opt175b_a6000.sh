export USE_TF=0 
BASE_LOG_DIR=~/experiments/zero_inference/
MODEL_NAME="opt-175b"
FULL_MODEL_NAME="facebook/${MODEL_NAME}"
QB=4

OFFLOAD_DIR=/local_nvme/zero_offload
mkdir -p $OFFLOAD_DIR

# BSZ=8
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR

# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk.txt 
# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk_q${QB}.txt 

# BSZ=32
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk_kv.txt
# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --kv-offload  --quant_bits ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk_q${QB}_kv.txt
# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --kv-offload --pin_kv_cache &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk_kv_pkv.txt
# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --disk-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --kv-offload --pin_kv_cache --async_kv_offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_disk_kv_pkv_akv.txt


# BSZ=24
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} --kv-offload &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}_kv.txt


# BSZ=16
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}.txt

BSZ=8
LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
mkdir -p  $LOG_DIR
deepspeed --num_gpus 1 run_model.py --dummy --model ${FULL_MODEL_NAME} --batch-size ${BSZ} --cpu-offload --gen-len 32 --pin-memory 0 --offload-dir ${OFFLOAD_DIR} --quant_bits ${QB} &> $LOG_DIR/ds_${MODEL_NAME}_bs${BSZ}_cpu_q${QB}.txt



# # 175b flexgen with compute schedule or partial offloading
# BSZ=64
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# OFFLOAD_DIR=/local_nvme/flexgen_offload
# mkdir -p $OFFLOAD_DIR
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_disk.txt
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 0 0 100 0 100 --gpu-batch-size ${BSZ} --offload-dir ${OFFLOAD_DIR} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_ccc_disk.txt  

# 175b flexgen with 4-bit weight quantization but without compute schedule or partial offloading
# BSZ=32
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_q4.txt    
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_ccc_q4.txt    

# BSZ=40
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_q4.txt    
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --cpu-cache-compute --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_ccc_q4.txt    


# BSZ=8
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_q4.txt    


# BSZ=16
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 100 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 --compress-weight  &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_cpu_q4.txt    

# # 30b flexgen with 4-bit weight quantization but without compute schedule or partial offloading
# BSZ=8
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 0 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_disk.txt    


# BSZ=16
# LOG_DIR=$BASE_LOG_DIR/${MODEL_NAME}_bs${BSZ}
# mkdir -p  $LOG_DIR
# python -m flexgen.flex_opt --model ${FULL_MODEL_NAME} --path _DUMMY_ --percent 0 0 100 0 100 0 --gpu-batch-size ${BSZ} --pin-weight 0 --num-gpu-batches 1 &> $LOG_DIR/fg_${MODEL_NAME}_bs${BSZ}_disk.txt    
