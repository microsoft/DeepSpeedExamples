#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path_finetune output/check_base \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --num_beams 5 \
    --num_beam_groups 5\
    --top_k 4 \
    --penalty_alpha 0.6 \
    --num_return_sequences 1 \
    --max_new_tokens 100
