#!/bin/bash

export HL_DATASET_PATH="Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets"
export HL_GBS=128
export HL_LEARNING_RATE=0.
export HL_LORA_LEARNING_RATE=1.1e-2
export HL_WEIGHT_DECAY=0.1
export HL_LORA_DIM=128
export HL_DROPOUT=0.1
export HL_EPOCHS=4
./train_step1_bloom_1.1b.sh
