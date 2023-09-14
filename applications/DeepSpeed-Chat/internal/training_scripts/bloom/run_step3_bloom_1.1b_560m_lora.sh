#!/bin/bash

export HL_DATASET_PATH="Dahoas/rm-static"
export HL_ACTOR_CP_ACT=0
export HL_CRITIC_CP_ACT=0
export HL_ACTOR_ZERO_STAGE=1
export HL_CRITIC_ZERO_STAGE=1
export HL_MBS=2
export HL_GBS=64
export HL_HYBRID_ENGINE=0
export HL_ACTOR_LR=0.0
export HL_LORA_ACTOR_LR=4e-4
export HL_ACTOR_WD=0.1
export HL_CRITIC_LR=0.0
export HL_LORA_CRITIC_LR=6e-4
export HL_CRITIC_WD=0.1
export HL_LORA_DIM=128
export HL_ACTOR_DROPOUT=0.0
export HL_CRITIC_DROPOUT=0.0
export HL_EPOCHS=1
export HL_NUM_WARMUP_STEPS=100
export HL_PRINT_ANSWERS_INTERVAL=0
export HL_SEED=${HL_SEED:=1}
./train_step3_bloom_1.1b_560m.sh
