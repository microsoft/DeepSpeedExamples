#!/bin/bash

hostfile=/job/hostfile
hosts=`cat $hostfile | awk '{print $1}' | paste -sd "," -`
PWD=`pwd`

declare -a configs=(
                    "32 1e-5 5" \
                    "32 3e-5 5" \
                    "32 5e-5 5" \                    
                    "32 7e-5 5" \
                    "32 9e-5 5" \
                    "32 1e-4 5" \
                    "32 2e-4 5" \
                    )

checkpoint_id="100"
JOBNAME="bert_base_progressive_layer_dropping_preLN_depth12_fp16_seq128_adam_chkpt${checkpoint_id}"
CHECKPOINT_PATH="~/workspace/dev/DeepSpeed-PLD/DeepSpeedExamples/bing_bert/bert_model_outputs/saved_models/adam_4k_seq128_progressive_layer_drop/epoch100_step65480/mp_rank_00_model_states.pt"
CHECKPOINT_PATH="$(echo -e "${CHECKPOINT_PATH}" | tr -d '[:space:]')" 

port=12345
iter=0
for config in "${configs[@]}"; do
    echo " $iter times"
    echo $config
    CONFIG_NAME="$(echo -e "${config}" | tr -d '[:space:]' | sed -r 's/[-]+/_/g')"

    cur_port=$((port + iter))
    echo " $cur_port cur_port"
    export PDSH_RCMD_TYPE=ssh
    cmd="pdsh -w ${hosts} cd $PWD; CUDA_VISIBLE_DEVICES=${iter} bash run_glue_bert_base_finetune.sh ${cur_port} '${config}' ${JOBNAME} ${CHECKPOINT_PATH} &"
    echo ${cmd}
    eval ${cmd}
    # pdsh -w ${hosts} "cd $PWD;cmd" &
    iter=$((iter + 1))
done

