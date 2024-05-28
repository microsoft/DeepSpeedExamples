# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

PROMPT_LENGTH=${PROMPT_LENGTH:-500}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-500}
ENABLE_PREFIX_CACHE=${ENABLE_PREFIX_CACHE:-false}
OUT_DIR=${OUT_DIR:-"results"}

ARG_ENABLE_PREFIX_CACHE=""
if [ "$ENABLE_PREFIX_CACHE" = "true" ]; then
    ARG_ENABLE_PREFIX_CACHE="--enable_prefix_cache"
fi

# Run benchmark
python ./run_benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --tp_size 1 \
        --num_replicas 1 \
        --max_ragged_batch_size 768 \
        --mean_prompt_length ${PROMPT_LENGTH} \
        --mean_max_new_tokens ${MAX_NEW_TOKENS} \
        --stream \
        --backend fastgen \
        --out_json_dir ${OUT_DIR} \
        --overwrite_results \
        ${ARG_ENABLE_PREFIX_CACHE}


# ### Gernerate the plots
python ./src/plot_th_lat.py

# echo "Find figures in ./plots/ and log outputs in ./results/"
