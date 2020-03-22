#! /bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
tag="notag"
if [ -z $1 ]; then
   config_json="$script_dir/ds_config.json"
else
   config_json="$script_dir/$1"
fi

NUM_GPUS_PER_NODE=${DLWS_NUM_GPU_PER_WORKER}

cat ${script_dir}/exp.dat | while read line
do
    #Model Parallelism and Data Parallelism
    num_nodes=`echo $line | awk '{print $1}'`
    mp_size=`echo $line | awk '{print $2}'`
    
    #If the number of nodes is not 1-100 then skip
    #takes care of commented lines, white spaces etc
    if [ "${num_nodes}" -ge 1 ] && [ "${num_nodes}" -le 100 ] ; then
        #Model Config
        nlayers=`echo $line | awk '{print $3}'`
        hidden_size=`echo $line | awk '{print $4}'`
        attn_heads=`echo $line | awk '{print $5}'`
        
        #Batch Size
        batch_size=`echo $line | awk '{print $6}'`
        
        #ZeRO Configs
        stage=`echo $line | awk '{print $7}'`
        reduce_scatter=`echo $line | awk '{print $8}'`
        contigious_gradients=`echo $line | awk '{print $9}'`
        rbs=`echo $line | awk '{print $10}'`
        agbs=`echo $line | awk '{print $11}'`
        
        #Actication Checkpointing and Contigious Memory
        PA=`echo $line | awk '{print $12}'`
        PA_CPU=`echo $line | awk '{print $13}'`
        CC=`echo $line | awk '{print $14}'`
        SYNCHRONIZE=`echo $line | awk '{print $15}'`
        PROFILE=`echo $line | awk '{print $16}'`
        
        chkp_layers=`echo $line | awk '{print $17}'`
        
        #Train Iterations
        train_iters=`echo $line | awk '{print $18}'`
        
        echo "Starting new experiement: $line"
        optimizations="s${stage}_rs-${reduce_scatter}_rbs${rbs}_agbs${agbs}_PA-${PA}_CPU-${PA-CPU}_CC-${CC}_SYNC-${SYNCHRONIZE}"
        deepscale_logfile="ds_${num_nodes}_${mp_size}_${batch_size}_${nlayers}_${hidden_size}_${attn_heads}_${chkp_layers}_${train_iters}"
        deepscale_logfile="${deepscale_logfile}_${optimizations}_${tag}.log"
        
        if [ -f $deepscale_logfile ]; then
            echo "skipping $deepscale_logfile"
        else
        
            gpt_options=" \
                --model-parallel-size ${mp_size} \
                --num-layers ${nlayers} \
                --hidden-size ${hidden_size} \
                --num-attention-heads ${attn_heads} \
                --batch-size ${batch_size} \
                --seq-length 1024 \
                --max-position-embeddings 1024 \
                --train-iters ${train_iters} \
                --resume-dataloader \
                --train-data webtext \
                --lazy-loader \
                --tokenizer-type GPT2BPETokenizer \
                --split 949,50,1 \
                --distributed-backend nccl \
                --lr 0.00015 \
                --no-load-optim \
                --lr-decay-style cosine \
                --weight-decay 1e-2 \
                --clip-grad 1.0 \
                --warmup .01 \
                --fp16
            "
            deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

            if [ "${contigious_gradients}" = "true" ]; then
                deepspeed_options="${deepspeed_options} \
                                --zero-contigious-gradients"
            fi

            chkp_opt=" \
                --checkpoint-activations \
                --checkpoint-num-layers ${chkp_layers}"

            if [ "${PA}" = "true" ]; then
                chkp_opt="${chkp_opt} \
                        --partition-activations"
            fi

            if [ "${PA_CPU}" = "true" ]; then
                chkp_opt="${chkp_opt} \
                        --checkpoint-in-cpu"
            fi

            if [ "${SYNCHRONIZE}" = "true" ]; then
                chkp_opt="${chkp_opt} \
                        --synchronize-each-layer"
            fi

            if [ "${CC}" = "true" ]; then
                chkp_opt="${chkp_opt} \
                        --contigious-checkpointing"
            fi
            
            if [ "${PROFILE}" = "true" ]; then
                chkp_opt="${chkp_opt} \
                        --profile-backward"
            fi
            
            full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

            run_cmd="deepspeed.pt --num_nodes ${num_nodes} --num_gpus ${NUM_GPUS_PER_NODE} pretrain_gpt2.py ${@:2} ${full_options} &> ${deepscale_logfile}"
            echo ${run_cmd}
            eval ${run_cmd}
        fi
        echo "experiment done"
    else
        echo "\ Skipping commented line: ${line} \ "
    fi
done
set +x
