# deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
#     --model_type=gpt2 \
#     --model_name_or_path=gpt2-large \
#     --sample_input sample_query.txt \
#     --fp16 \
#     --ds-inference

# deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
#     --model_type=gptneo \
#     --model_name_or_path=Eleu-therAI/gpt-neo-2.7B \
#     --sample_input sample_query.txt \
#     --fp16 \
#     --trt

deepspeed --num_nodes 1 --num_gpus 1 run_generation.py \
    --model_type=gptj \
    --model_name_or_path=EleutherAI/gpt-j-6B \
    --sample_input sample_query.txt \
    --fp16 \
    --trt