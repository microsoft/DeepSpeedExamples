## 🦙 Finetune LLaMA on 2080Ti

For a long time, it's believed that RTX3090 (~24 GB memory) is the entry threshold for to train and fine-tune LLaMA and other large language models.

In this project, we use 2080Ti to fine-tune LLaMA (tested up to LLaMA-30B), and we believe it can also run on lower devices such as RTX3060 8G. With 8 2080Tis, we can train 1-epoch of alpaca-52k dataset within 1.5 hours.

## 🥪 Prepare

### Requirements
- deepspeed >= 0.8.3
- transformers >= 4.28.0dev
- peft >= 0.3.0dev

### Pre-trained model and datasets

1. Run `download_llama.py` to download pretrained LLaMA weights from 🤗 [HuggingFace](https://huggingface.co/decapoda-research).
2. Download Alpaca instruction fine-tuning dataset from [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora), including
   - `alpaca_data_cleaned_archive.json`
   - `alpaca_data_gpt4.json`

## 🔆 Start Training

Assume pretrained llama weights are stored at `/path/to/huggingface/decapoda-research/llama-7b-hf`, and alpaca datasets are at `/path/to/alpaca/alpaca_data_*.json`, simply run command

```bash
num_gpus=$(nvidia-smi -L | wc -l)

deepspeed --num_gpus=${num_gpus} \
    --module alpaca.train 7b \
    --batch-size=8 \
    --deepspeed_config ds_config.json \
    --load-pretrain \
    --pretrained-dir=/path/to/huggingface \
    --alpaca-data-dir=/path/to/alpaca \
    --checkpoint-root-dir=./checkpoints/alpaca
```

## 💡 How we did it

We use multiple techniques combined to save GPU memory, including:
- **LoRA**: reduce optimization overhead
- **float16 mixed-precision**: save 50% parameter memory
- **ZeRO stage-3 + CPU offloading**: reduce parameter memory to single-layer
- **Gradient checkpointing**: save activation cache, increase per-GPU microbatch size from 1 to 16

Although some training frameworks such as [accelerate](https://github.com/huggingface/accelerate) and [fabric](https://lightning.ai/docs/fabric/) are supposed to support these techniques, they failed to work when all these techniques are enabled.

In this project, we use vanilla deepspeed engine with some custom utility functions to support the training. Please check the code for more details.