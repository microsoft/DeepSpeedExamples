#!/bin/bash

export HL_DATASET_PATH="Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets"
./train_step2_bloom_560m.sh
