# --num_gpus 1
deepspeed -i localhost:0 main.py \
	--model_name_or_path facebook/opt-1.3b \
	--gradient_accumulation_steps 1 \
	--per_device_train_batch_size 2 \
	--zero_stage 2 \
	--deepspeed \
	--lora_dim 0 \
	--output_dir ./output
#--gradient_checkpointing \
