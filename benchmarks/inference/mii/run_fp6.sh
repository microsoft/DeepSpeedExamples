# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export TRANSFORMERS_CACHE=/home/blob/
export HF_HOME=/home/blob/
export HF_DATASETS_CACHE=/home/blob/
MODELS=(NousResearch/Llama-2-13b-hf)

for MODEL in ${MODELS[@]}; do
    python ./run_benchmark.py --model ${MODEL} --num_requests 128 --stream --backend fastgen --fp6
done