#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

ACTOR_MODEL_PATH="AdamG012/chat-opt-1.3b-sft-deepspeed"
CRITIC_MODEL_PATH="AdamG012/chat-opt-350m-reward-deepspeed"

#for z in {2..3}
for z in 3
do
    #for he in true false
    for he in true
    do
        #for offload in true false
        for offload in false
        do
            #for lora in true false
            for lora in true
            do
                #for offload_reference_model in true false
                for offload_reference_model in true
                do
                    for gradient_checkpointing in true
                    do
                        echo "CUDA_LAUNCH_BLOCKING"
                        echo $CUDA_LAUNCH_BLOCKING
                        cmd="bash training_scripts/single_node/run_1.3b_lora_sage_offl.sh \
                            $ACTOR_MODEL_PATH \
                            $CRITIC_MODEL_PATH \
                            ${z} \
                            ${z} \
                            ${he} \
                            ${offload} \
                            ${lora} \
                            ${offload_reference_model} \
                            ${gradient_checkpointing} \
                            z${z}_he_${he}_lora_${lora}_orm_${offload_reference_model}_gc_${gradient_checkpointing}_REPEAT"
                        echo "----------------------------- CALLING SHELL SCRIPT -----------------------------"
                        echo $cmd
                        $cmd
                        pkill -9 python
                        sleep 60
                        echo ""

                        export CUDA_LAUNCH_BLOCKING=1
                        echo "CUDA_LAUNCH_BLOCKING"
                        echo $CUDA_LAUNCH_BLOCKING
                        cmd="bash training_scripts/single_node/run_1.3b_lora_sage_offl.sh \
                            $ACTOR_MODEL_PATH \
                            $CRITIC_MODEL_PATH \
                            ${z} \
                            ${z} \
                            ${he} \
                            ${offload} \
                            ${lora} \
                            ${offload_reference_model} \
                            ${gradient_checkpointing} \
                            z${z}_he_${he}_lora_${lora}_orm_${offload_reference_model}_gc_${gradient_checkpointing}_REPEAT_CUDA_BLOCKING"
                        echo "----------------------------- CALLING SHELL SCRIPT -----------------------------"
                        echo $cmd
                        $cmd
                        pkill -9 python
                        sleep 60
                        echo ""
                    done
                done
            done
        done
    done
done
