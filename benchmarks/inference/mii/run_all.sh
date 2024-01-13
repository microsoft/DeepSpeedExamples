MODELS=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-70b-hf tiiuae/falcon-180B)

for MODEL in ${MODELS[@]}; do
    python run_benchmark.py --model ${MODEL} --use_defaults --stream
    python run_benchmark.py --model ${MODEL} --use_defaults --stream --vllm
done
