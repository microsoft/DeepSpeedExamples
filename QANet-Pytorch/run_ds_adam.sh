#!/bin/bash

# Path to processed data
data_dir=/data/QANet/processed

if [ ! -d ${data_dir} ]; then
    echo "Processed data directory does not exist, please create it first!"
    exit 1
fi

squad_dir=${data_dir}/SQuAD
squad_args="--train_processed_data ${squad_dir}/train-v1.1-processed.pkl \
            --dev_processed_data ${squad_dir}/dev-v1.1-processed.pkl \
            --train_meta_data ${squad_dir}/train-v1.1-meta.pkl \
            --train_eval_data ${squad_dir}/dev-v1.1-meta.pkl \
            --dev_eval_data ${squad_dir}/dev-v1.1-eval.pkl \
            --processed_word_embedding ${squad_dir}/word_emb.pkl \
            --word_dictionary ${squad_dir}/word_dic.pkl \
            --processed_char_embedding ${squad_dir}/char_emb.pkl \
            --char_dictionary ${squad_dir}/char_dic.pkl \
           "

deepspeed_options="--deepspeed --deepspeed_config short_deepspeed_adam.json"
args="--data_processed ${squad_args} --with_cuda --use_ema ${deepspeed_options}"
run_cmd="deepspeed.pt --num_nodes 1 --num_gpus 16 main_ds_adam.py $@ ${args}"
echo ${run_cmd}
eval ${run_cmd}
