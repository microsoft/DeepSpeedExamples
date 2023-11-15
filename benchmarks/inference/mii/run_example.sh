Run the code

RAGGED_BATCH_SIZE=768
PARAM_SIZES=(7b)
DEPLOYMENT_NAME=llama2-7b-tp1-b768
python server.py --model_name meta-llama/Llama-2-7b-hf -d llama2-7b-tp1-b768 -m 1 -b 768 start

DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=2600 MAX_NEW_TOKENS=60 bash ./run_benchmark_client.sh
DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=2600 MAX_NEW_TOKENS=128 bash ./run_benchmark_client.sh
DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=1200 MAX_NEW_TOKENS=60 bash ./run_benchmark_client.sh
DEPLOYMENT_NAME=${DEPLOYMENT_NAME} PROMPT_LENGTH=1200 MAX_NEW_TOKENS=128 bash ./run_benchmark_client.sh

echo "Stopping server"
python server.py -d ${DEPLOYMENT_NAME} stop
sleep 120

### Plots
python plot_th_lat.py --log_dir . --test --no_vllm
python plot_effective_throughput.py --log_dir . --test --no_vllm

echo "Find the plots in the charts directory and the logs inside logs.llama2-7b-tp1-b768"