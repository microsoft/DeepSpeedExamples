deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --sample_input sample_query.txt \
    --fp16 \
    --ds-inference
