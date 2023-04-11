import os
import time
from typing import List, Dict, Tuple
from pathlib import Path

import refile
import hjson
import click
import torch
import peft
import deepspeed
import transformers

from torch import nn
from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage
from transformers import LlamaForCausalLM, AutoConfig

from tqdm import tqdm
from loguru import logger

from .dsutils import (
    trainable_parameters_stat,
    zero3_load_pretrained, zero3_save_lora_model
)
from .dataset import make_dataset, AlpacaCollectionEnum, LLamaTokenizer


@click.command()
@click.argument("model_size", type=click.Choice(["7b", "13b", "30b"]))
@click.option("--context-window-size", type=int, default=512)
@click.option("--batch-size", type=int, default=-1)
@click.option("--alpaca-data-dir", type=str, default="/data/alpaca")
@click.option(
    "-c", "--collection", "alpaca_collections", multiple=True,
    type=click.Choice([e.value for e in AlpacaCollectionEnum]),
    default=[AlpacaCollectionEnum.Alpaca.value],
)
@click.option("--pretrained-dir", default="/data/huggingface/")
@click.option("--load-pretrain/--no-load-pretrain", "load_pretrain", default=False)
@click.option("--checkpoint-root-dir", default="./checkpoints/")
@click.option("--gradient-checkpoint/--no-gradient-checkpoint", "enable_gradient_checkpoint", default=True)
@click.option("--dry-run/--no-dry-run", "is_dry_run", default=False)
@click.option("--local-rank", "--local_rank", default=-1, type=int)
@click.option("--deepspeed_config", "ds_config_file", default="", type=str)
def main(
    model_size: str, context_window_size: int, batch_size: int,
    alpaca_data_dir: str, alpaca_collections: Tuple[str],
    pretrained_dir: str, checkpoint_root_dir: str, load_pretrain: bool = False,
    enable_gradient_checkpoint: bool = True, is_dry_run: bool = False,
    local_rank: int = -1, ds_config_file: str = ""
):
    assert (torch.cuda.is_available())

    # --- Init configurations ---
    logger.configure(extra={'rank': str(local_rank)})
    logger.configure(handlers=[dict(
        sink=lambda msg: tqdm.write(msg, end=''),
        format="[<blue>R-{extra[rank]}</blue>] [<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>] [<level>{level}</level>] {message}",
        colorize=True
    )])
    device = torch.device("cuda", local_rank) if (local_rank > -1) else torch.device('cuda')

    with open(ds_config_file) as f:
        ds_config = hjson.load(f)
        if batch_size > 0:
            ds_config['train_micro_batch_size_per_gpu'] = batch_size
        else:
            batch_size = ds_config['train_micro_batch_size_per_gpu']

    pretrained_model_dir = os.path.join(pretrained_dir, f"decapoda-research/llama-{model_size}-hf")
    checkpoint_dir = os.path.join(checkpoint_root_dir, f"llama-{model_size}")

    if is_dry_run:
        logger.warning("This is a DRY run")

    # --- Load pretrained model ---
    logger.info(f"Loading model from {pretrained_model_dir}, load pretrain weights: {load_pretrain}")
    zero_init_opts = dict(
        config_dict_or_path=ds_config,
        remote_device="cpu",
        dtype=torch.float16,
        enabled=ds_config.get('zero_optimization', {}).get('stage', -1) == 3,
    )
    with deepspeed.zero.Init(**zero_init_opts):
        model_cfg = AutoConfig.from_pretrained(pretrained_model_dir)
        model = LlamaForCausalLM(model_cfg)

    if load_pretrain:
        logger.info("Loading pretrained weights")
        zero3_load_pretrained(model, pretrained_model_dir)
        logger.info("pretrained weights loaded")

    # config LoRA finetune
    if enable_gradient_checkpoint:
        model.enable_input_require_grads()     # required for gradient checkpointing
        model.gradient_checkpointing_enable()  # enable gradient checkpointing
    lora_config = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0, bias="none", task_type="CAUSAL_LM")
    model = peft.get_peft_model(model, lora_config)
    logger.info("Model created.")

    see_memory_usage("After zero init, sleep 10 seconds for debugging", force=True)
    time.sleep(10)

    # Load dataset
    tokenizer = LLamaTokenizer(os.path.join(pretrained_model_dir, "tokenizer.model"))
    train_ds = make_dataset(alpaca_data_dir, alpaca_collections, tokenizer, block_size=context_window_size)

    engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        lr_scheduler=lambda optimizer: transformers.get_scheduler(
            transformers.SchedulerType.CONSTANT_WITH_WARMUP,
            optimizer,
            num_warmup_steps=100,
        ),
        training_data=train_ds,
        config=ds_config,
    )
    engine: deepspeed.DeepSpeedEngine
    logger.info("DeepSpeed engine created")
    logger.info(f"World Size: {dist.get_world_size()}, Local Rank: {dist.get_local_rank()}")

    pstat = trainable_parameters_stat(engine)
    logger.info(
        f"trainable params: {pstat['trainable']} || all params: {pstat['total']} || trainable%: {100 * pstat['trainable'] / pstat['total']}"
    )

    see_memory_usage("After deepspeed init, sleep 10 seconds for debugging", force=True)
    time.sleep(10)

    global_step = 0
    for epoch in range(10):
        if not is_dry_run:
            epoch_save_dir = os.path.join(checkpoint_dir, f"epoch-{epoch:04d}")
            zero3_save_lora_model(model, epoch_save_dir)

        engine.train()
        pbar = tqdm(train_loader, dynamic_ncols=True, desc=f"Epoch: {epoch:04d}", disable=(not local_rank == 0))
        for bidx, batch in enumerate(pbar):

            batch = {k: v.to(device) for k, v in batch.items()}
            batch['labels'] = batch['input_ids'].clone()
            # labels are shifted **inside** the model
            with torch.cuda.amp.autocast(cache_enabled=False):
                outputs = engine.forward(**batch, use_cache=False)

            loss = outputs.loss
            engine.backward(loss)
            engine.step()

            if local_rank == 0:
                mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
                mem_info = f"({mem_alloc:.2f}/{mem_reserved:.2f}"
                logger.info(f"Clock: {epoch},{bidx}/{len(train_loader)} | Loss: {loss.item()}, LR: {lr_scheduler.get_last_lr()[0]:.4g}, Mem: {mem_info}")

            global_step += 1

            if global_step % 100 == 0 and not is_dry_run:
                torch.cuda.empty_cache()
                checkpoint_save_dir = os.path.join(checkpoint_dir, f"checkpoint-step{global_step}")
                zero3_save_lora_model(model, checkpoint_save_dir)


if __name__ == "__main__":
    main()