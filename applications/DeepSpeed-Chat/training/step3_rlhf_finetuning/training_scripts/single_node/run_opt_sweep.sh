#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

ACTOR_MODEL_PATH="AdamG012/chat-opt-1.3b-sft-deepspeed"
CRITIC_MODEL_PATH="AdamG012/chat-opt-350m-reward-deepspeed"

for z in {2..3}
do
    for he in true false
    do
        for offload in true false
        do
            for base in True False
            do
                if [ $he == 'false' ] && [ $base == 'True' ]; then
                    continue;
                fi
                cmd="bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH ${z} ${z} ${he} ${offload} false ${base} z${z}_he_${he}_offload_${offload}_base_${base}_sft"
				echo $cmd
                $cmd
				pkill -9 python
				sleep 60
            done
        done
    done
done
