### Run the server
echo "Starting the server"
python server.py \
        --model meta-llama/Llama-2-7b-hf \
        --tp_size 1 \
        --max_ragged_batch_size 768 \
        start

### This command will run the client with 60 generation steps and input prompt length of 2600
echo "Running the client"
python client.py \
        --mean_prompt_length 2600 \
        --mean_max_new_tokens 60

### Stop the server
echo "Stopping the server"
python server.py stop
sleep 120

### Gernerate the plots
python plot_th_lat.py

echo "Find figures in ./plots/ and log outputs in ./results/"
