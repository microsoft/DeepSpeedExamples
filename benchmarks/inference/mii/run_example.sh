### Run the server
python server.py \
        --model meta-llama/Llama-2-7b-hf \
        --deployment_name llama2-7b-tp1-b768 \
        --tp_size 1 \
        --max_ragged_batch_size 768 \
        start

### This command will run the client with 60 generation steps and input prompt length of 2600
python client.py \
        --deployment_name llama2-7b-tp1-b768 \
        --mean_prompt_length 2600 \
        --mean_max_new_tokens 60

### Stop the server
echo "Stopping server"
python server.py -d ${DEPLOYMENT_NAME} stop
sleep 120

### Gernerate the plots
python plot_th_lat.py --log_dir . --test --no_vllm
python plot_effective_throughput.py --log_dir . --test --no_vllm

echo "Find the plots in the charts directory and the logs inside logs.llama2-7b-tp1-b768"
