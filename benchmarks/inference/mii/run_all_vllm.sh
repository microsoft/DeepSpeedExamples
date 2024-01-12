RAGGED_BATCH_SIZE=768
PARAM_SIZES=(7b 13b 70b)

declare -A TP_SIZES
TP_SIZES["7b"]="1"
TP_SIZES["13b"]="1:2:4"
TP_SIZES["70b"]="4:8"

# model dependent parameters
# LLAMA-2
MAX_PROMPT_LENGTH=4000
PROMPT_LENGTH_LIST=(2600 1200)
MAX_NEW_TOKENS_LIST=(60 128)

# # Falcon
# MAX_PROMPT_LENGTH=2000
# PROMPT_LENGTH_LIST=(1900 1200)
# MAX_NEW_TOKENS_LIST=(60 128)

for PARAM_SIZE in ${PARAM_SIZES[@]}; do
    MODEL_NAME=meta-llama/Llama-2-${PARAM_SIZE}-hf
    # MODEL_NAME=tiiuae/falcon-${PARAM_SIZE}

    IFS=':' read -ra TP_VALUES <<< ${TP_SIZES[${PARAM_SIZE}]}
    for TP in ${TP_VALUES[@]}; do
        DEPLOYMENT_NAME=vllm-llama2-${PARAM_SIZE}-tp${TP}
        # DEPLOYMENT_NAME=falcon-${PARAM_SIZE}-tp${TP}-b${RAGGED_BATCH_SIZE}

        echo "Starting server"
        python -m vllm.entrypoints.api_server --host 127.0.0.1 --port 26500 --tensor-parallel-size ${TP} --model ${MODEL_NAME} &
        sleep 60

        for PROMPT_LENGTH in ${PROMPT_LENGTH_LIST[@]}; do
            for MAX_NEW_TOKENS in ${MAX_NEW_TOKENS_LIST[@]}; do
                VLLM="--vllm"
                source ./run_benchmark_client.sh
            done
        done

        echo "Stopping server"
        pkill -u ${USER} -f vllm.entrypoints.api_server
        sleep 30
    done
done
