#!/bin/bash

export HL_DATASET_PATH="Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets"
export HL_CRITIC_ZERO_STAGE=1
export HL_MBS=8
export HL_GBS=256
export HL_CRITIC_MODEL=bigscience/bloom-560m
export HL_LEARNING_RATE=2e-5
export HL_LORA_LEARNING_RATE=5e-3
export HL_WEIGHT_DECAY=0.1
export HL_EPOCHS=3
export HL_LORA_DIM=128
export HL_DROPOUT=0.0
./train_step2_bloom_560m.sh
