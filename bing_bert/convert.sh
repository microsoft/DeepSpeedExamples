# Convert Tensorflow checkpoint to DeepSpeed
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    convert_bert_ckpt_to_deepspeed.py \
    --ckpt_type TF \
    --ckpt_path ~/test/uncased_L-24_H-1024_A-16/bert_model.ckpt \
    --bert_config_file ~/test/uncased_L-24_H-1024_A-16/bert_config.json \
    --deepspeed_dump_dir ~/test/deepspeed_ckpt \
    --deepspeed_config deepspeed_bsz64k_lamb_config_seq128.json

# Convert Huggingface checkpoint to DeepSpeed
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    convert_bert_ckpt_to_deepspeed.py \
    --ckpt_type HF \
    --ckpt_path ~/test/huggingface/bert-large-uncased-whole-word-masking-pytorch_model.bin \
    --bert_config_file /home/eltonz/test/huggingface/config.json \
    --deepspeed_dump_dir ~/test/huggingface/deepspeed_ckpt \
    --deepspeed_config deepspeed_bsz64k_lamb_config_seq128.json
