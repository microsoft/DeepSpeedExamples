#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

for z in 2
do
    for offload in true false
    do
        cmd="bash training_scripts/single_node/run_1.3b.sh ${z} ${offload} step1_z${z}_offload_${offload}_lora_false"
        echo $cmd
        $cmd
        pkill -9 python
        sleep 60

        cmd_lora="bash training_scripts/single_node/run_1.3b_lora.sh ${z} ${offload} step1_z${z}_offload_${offload}_lora_true"
        echo $cmd_lora
        $cmd_lora
        pkill -9 python
        sleep 60
    done
done
