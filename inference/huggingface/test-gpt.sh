test=$1
# input="sample_query.txt"
input="single_query.txt"
name="bigscience/T0"
type="t5"
name=gpt2-xl
type=gpt2
#name="facebook/opt-30b"
#type=opt
if [ $test == "hf" ]; then
    deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
        --model_type=$type \
        --model_name_or_path=$name\
        --fp16 \
        --sample_input $input
elif [ $test == "ds" ]; then
    deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
        --model_type=$type \
        --model_name_or_path=$name \
        --sample_input $input \
        --fp16 \
        --ds-inference
elif [ $test == "gpu" ]; then
    deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
        --model_type=$type \
        --model_name_or_path=$name \
        --sample_input $input \
        --fp16 \
        --ds-zero-inference \
        --ds_config_path=ds_config_gpu.json
elif [ $test == "cpu" ]; then
    deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
        --model_type=$type \
        --model_name_or_path=$name \
        --sample_input $input \
        --fp16 \
        --ds-zero-inference \
        --ds_config_path=ds_config_cpu.json
else
    deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
        --model_type=$type \
        --model_name_or_path=$name \
        --sample_input $input \
        --fp16 \
        --ds-zero-inference \
        --ds_config_path=ds_config_nvme.json
fi