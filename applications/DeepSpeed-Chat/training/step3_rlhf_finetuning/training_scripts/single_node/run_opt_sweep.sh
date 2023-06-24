#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

ACTOR_MODEL_PATH="AdamG012/chat-opt-1.3b-rlhf-actor-deepspeed"
CRITIC_MODEL_PATH="AdamG012/chat-opt-350m-reward-deepspeed"

#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 2 2 true true 2_2_true_true
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 true true 3_3_true_true
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 false false 3_3_false_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 true false 3_3_true_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 false true 3_3_false_true
#pkill -9 python
#sleep 60

#------------------ ds-chat sweep ------------------

#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 2 2 false false false 2_2_false_false_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 2 2 false true false 2_2_false_true_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 2 2 false false false 2_2_true_false_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 2 2 true false false 2_2_true_false_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 2 2 true true false 2_2_true_true_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 false false false 3_3_false_false_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 false true false 3_3_false_true_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 false false false true 3_3_true_false_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 true false false 3_3_true_false_false
#pkill -9 python
#sleep 60
#
#bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH 3 3 true true false 3_3_true_true_false
#pkill -9 python
#sleep 60

#--------------------------------------------------------------------------------------------------------------------

for z in {2..3}
do
    for he in true false
    do
        for offload in true false
        do
            for base in True False
            do
                if [ $he == 'false' ] && [ $base == 'False' ]; then
                    continue;
                fi
                cmd="bash training_scripts/single_node/run_1.3b.sh $ACTOR_MODEL_PATH $CRITIC_MODEL_PATH ${z} ${z} ${he} ${offload} false ${base} z${z}_he_${he}_offload_${offload}_base_${base}"
				echo $cmd
                $cmd
				pkill -9 python
				sleep 60
            done
        done
    done
done
