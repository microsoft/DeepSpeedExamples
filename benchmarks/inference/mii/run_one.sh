###################################
# 7B
PARAM_SIZE=7b
TP=1
RAGGED_BATCH_SIZE=512
DEPLOYMENT_NAME=llama2-${PARAM_SIZE}-tp${TP}
python server_ragged_batch_llama2.py --model_name meta-llama/Llama-2-${PARAM_SIZE}-hf -d ${DEPLOYMENT_NAME} -m ${TP} -b ${RAGGED_BATCH_SIZE} start

DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=3072 MAX_NEW_TOKENS=60 bash ./run_bench_client_num.sh
