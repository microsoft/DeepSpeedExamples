# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export TRANSFORMERS_CACHE=/blob/transformers_cache
export HF_HOME=/blob/transformers_cache
export HF_DATASETS_CACHE=/blob/transformers_cache
#MODELS=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-70b-hf tiiuae/falcon-40B tiiuae/falcon-180B microsoft/phi-2 mistralai/Mixtral-8x7B-v0.1)
# MODELS=(meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-70b-hf tiiuae/falcon-40B tiiuae/falcon-180B microsoft/phi-2 mistralai/Mixtral-8x7B-v0.1)
# MODELS=(meta-llama/Llama-2-7b-hf)
MODELS=(NousResearch/Llama-2-13b-hf)
# MODELS=(mistralai/Mixtral-8x7B-v0.1)
# MODELS=(tiiuae/falcon-40B)
 
for MODEL in ${MODELS[@]}; do
    # python ./run_benchmark.py --model ${MODEL} --stream
    # python ./run_benchmark.py --model ${MODEL} --stream --vllm
    python3 ./run_benchmark.py --model ${MODEL} --stream --out_json_dir ./results1/FP16correct/
    # python ./run_benchmark.py --model ${MODEL} --stream --out_json_dir ./results/FP16/
    # python ./run_benchmark.py --model ${MODEL} --stream --vllm
done

# # Extra runs for Mixtral with non-default settings
# python ./run_benchmark.py --model mistralai/Mixtral-8x7B-v0.1 --stream --tp_size 4 --mean_prompt_length 500 --mean_max_new_tokens 150 500 1024
# python ./run_benchmark.py --model mistralai/Mixtral-8x7B-v0.1 --stream --tp_size 4 --mean_prompt_length 500 --mean_max_new_tokens 150 500 1024 --vllm