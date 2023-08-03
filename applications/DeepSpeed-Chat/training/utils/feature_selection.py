#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch

from deepspeed.utils import OnDevice
from transformers import AutoConfig, AutoModelForCausalLM

from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters


def print0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def feature_selection_step3(args, model_class=AutoModelForCausalLM):
    # create meta actor (trainable)
    meta_actor, actor_config = _create_meta_model(
        args.actor_model_name_or_path, model_class)
    if args.actor_lora_dim > 0:
        meta_actor = _apply_lora(meta_actor, args, args.actor_lora_dim,
                                 args.actor_lora_module_name)
    actor_memory = _memory_overhead(
        meta_model=meta_actor,
        meta_config=actor_config,
        zero_stage=args.actor_zero_stage,
        seq_len=args.max_prompt_seq_len,
        batch_size=args.per_device_train_batch_size,
        gradient_ckpt=args.actor_gradient_checkpointing,
        lora_dim=args.actor_lora_dim)

    # create meta critic (trainable)
    meta_critic, critic_config = _create_meta_model(
        args.critic_model_name_or_path, model_class)
    if args.critic_lora_dim > 0:
        meta_actor = _apply_lora(meta_actor, args, args.critic_lora_dim,
                                 args.critic_lora_module_name)
    critic_memory = _memory_overhead(
        meta_model=meta_critic,
        meta_config=critic_config,
        zero_stage=args.critic_zero_stage,
        seq_len=args.max_prompt_seq_len,
        batch_size=args.per_device_train_batch_size,
        gradient_ckpt=args.critic_gradient_checkpointing,
        lora_dim=args.critic_lora_dim)

    # meta reference model based on actor
    meta_ref, ref_config = _create_meta_model(args.actor_model_name_or_path,
                                              model_class)
    zero_stage = 3 if args.actor_zero_stage == 3 else 0
    ref_memory = 0
    if not args.offload_reference_model:
        ref_memory = _memory_overhead(
            meta_model=meta_ref,
            meta_config=ref_config,
            zero_stage=zero_stage,
            seq_len=args.max_prompt_seq_len,
            batch_size=args.per_device_train_batch_size,
            gradient_ckpt=True,
            trainable=False)
    else:
        # TODO(Cheng/Jeff): this uses zero-inference, what will the memory overhead be in this case?
        # this might be the overhead of a single model layer?
        ref_memory = 0

    # meta ema model based on reference model
    ema_memory = 0
    if args.enable_ema:
        ema_memory = _memory_overhead(
            meta_model=meta_ref,
            meta_config=ref_config,
            zero_stage=zero_stage,
            seq_len=args.max_prompt_seq_len,
            batch_size=args.per_device_train_batch_size,
            trainable=False,
            gradient_ckpt=True,
            dtype=torch.float)

    # create meta reward model based on critic
    meta_reward, reward_config = _create_meta_model(
        args.critic_model_name_or_path, model_class)
    zero_stage = 3 if args.actor_zero_stage == 3 else 0
    reward_memory = _memory_overhead(
        meta_model=meta_reward,
        meta_config=reward_config,
        zero_stage=zero_stage,
        seq_len=args.max_prompt_seq_len,
        batch_size=args.per_device_train_batch_size,
        gradient_ckpt=True,
        trainable=False)

    # TODO(Cheng/Jeff): add memory overhead for KV-cache overhead
    # https://github.com/microsoft/DeepSpeed/blob/d10b8ca011b18eba3a6ca56f4208a732d7fbb744/csrc/transformer/inference/includes/inference_context.h#L116-L152

    print0("-----------------------------------------")
    print0(f"** Total actor memory: {actor_memory:.2f} GB")
    print0(f"** Total critic memory: {critic_memory:.2f} GB")
    print0(f"** Total reference memory: {ref_memory:.2f} GB")
    print0(f"** Total ema memory: {ema_memory:.2f} GB")
    print0(f"** Total reward memory: {reward_memory:.2f} GB")
    print0("-----------------------------------------")
    total_memory = actor_memory + critic_memory + ref_memory + ema_memory + reward_memory
    print0(f"*** Total memory required: {total_memory:.2f} GB")
    print0("-----------------------------------------")


def _memory_overhead(meta_model,
                     meta_config,
                     zero_stage,
                     seq_len,
                     batch_size,
                     gradient_ckpt=False,
                     lora_dim=0,
                     trainable=True,
                     dtype=torch.half):
    GB = 1024**3
    world_size = torch.distributed.get_world_size()
    psize = 2 if (dtype == torch.half or dtype == torch.bfloat16) else 4

    trainable_params = sum(
        [p.numel() if p.requires_grad else 0 for p in meta_model.parameters()])
    frozen_params = sum([
        p.numel() if not p.requires_grad else 0
        for p in meta_model.parameters()
    ])
    total_params = trainable_params + frozen_params

    activation_mem_required = _activation_memory_estimate(
        meta_config, lora_dim, gradient_ckpt, seq_len, batch_size)

    if not trainable:
        model_memory = (total_params * psize) / GB
        return model_memory + activation_mem_required
    assert dtype != torch.float, "currently do not support fp32 trainable models"

    mem_per_gpu = 0
    if zero_stage == 0:
        mem_per_gpu = (total_params * 2 + trainable_params * 2 +
                       trainable_params * 16) / GB
    elif zero_stage == 1:
        mem_per_gpu = total_params * 2  # model weights
        mem_per_gpu += trainable_params * 2  # model grads
        mem_per_gpu += (
            trainable_params *
            (12 + 4)) / world_size  # sharded optim states + fp32 sharded grads
        mem_per_gpu /= GB
    elif zero_stage == 2:
        mem_per_gpu = total_params * 2  # model weights
        mem_per_gpu += (trainable_params *
                        2) / world_size  # model grads are sharded
        mem_per_gpu += (
            trainable_params *
            (12 + 4)) / world_size  # sharded optim states + fp32 sharded grads
        mem_per_gpu /= GB
    elif zero_stage == 3:
        mem_per_gpu = (total_params *
                       2) / world_size  # model weights are sharded
        mem_per_gpu += (trainable_params *
                        2) / world_size  # model grads are sharded
        mem_per_gpu += (
            trainable_params *
            (12 + 4)) / world_size  # sharded optim states + fp32 sharded grads
        mem_per_gpu /= GB

    return mem_per_gpu + activation_mem_required


def feature_selection(args, model_class):
    meta_model, model_config = _create_meta_model(args.model_name_or_path,
                                                  model_class)
    nparams = sum([p.numel() for p in meta_model.parameters()])
    print0(f"[pre-lora] num params: {nparams}")

    if args.lora_dim > 0:
        meta_model = _apply_lora(meta_model, args, args.lora_dim,
                                 args.lora_module_name)

    nparams = sum([p.numel() for p in meta_model.parameters()])
    print0(f"[post-lora] num params: {nparams}")

    # [pre-LoRA] num params: 1,315,758,080
    # [post-LoRA] num params: 1,429,004,288
    # LoRA adds 113,246,208 parameters

    trainable_params = sum(
        [p.numel() if p.requires_grad else 0 for p in meta_model.parameters()])
    frozen_params = sum([
        p.numel() if not p.requires_grad else 0
        for p in meta_model.parameters()
    ])
    print0(f"{trainable_params=}, {frozen_params=}")

    #[pre-lora] num params: 1315758080
    #[post-lora] num params: 1429004288
    #trainable_params=221,044,736
    #frozen_params=1,207,959,552

    GB = 1024**3
    world_size = torch.distributed.get_world_size()
    mem_per_gpu = torch.cuda.get_device_properties(0).total_memory / GB

    # model weights (fp16) + gradients (fp16) + optimizer states (fp16/fp32)
    z0_model_states_mem_required = (nparams * 2 + trainable_params * 2 +
                                    trainable_params * 16) / GB
    print0(
        f'[ZeRO=0] Total model/optim states required: {z0_model_states_mem_required} GB'
    )

    z1_model_states_mem_required = nparams * 2  # model weights
    z1_model_states_mem_required += trainable_params * 2  # model grads
    z1_model_states_mem_required += (
        trainable_params *
        (12 + 4)) / world_size  # sharded optim states + fp32 sharded grads
    z1_model_states_mem_required /= GB
    print0(
        f'[ZeRO=1] Total model/optim states required: {z1_model_states_mem_required} GB'
    )

    z2_model_states_mem_required = nparams * 2  # model weights
    z2_model_states_mem_required += (trainable_params *
                                     2) / world_size  # model grads are sharded
    z2_model_states_mem_required += (
        trainable_params *
        (12 + 4)) / world_size  # sharded optim states + fp32 sharded grads
    z2_model_states_mem_required /= GB
    print0(
        f'[ZeRO=2] Total model/optim states required: {z2_model_states_mem_required} GB'
    )

    z3_model_states_mem_required = (
        nparams * 2) / world_size  # model weights are sharded
    z3_model_states_mem_required += (trainable_params *
                                     2) / world_size  # model grads are sharded
    z3_model_states_mem_required += (
        trainable_params *
        (12 + 4)) / world_size  # sharded optim states + fp32 sharded grads
    z3_model_states_mem_required /= GB
    print0(
        f'[ZeRO=3] Total model/optim states required: {z3_model_states_mem_required} GB'
    )

    activation_mem_required = _activation_memory_estimate(
        model_config, args.lora_dim, False, args.max_seq_len,
        args.per_device_train_batch_size)
    activation_mem_required_gc = _activation_memory_estimate(
        model_config, args.lora_dim, True, args.max_seq_len,
        args.per_device_train_batch_size)

    print0(
        f"Estimated activation memory required without gradient checkpointing: {activation_mem_required} GB, with checkpointing: {activation_mem_required_gc} GB"
    )

    if args.gradient_checkpointing:
        activation_mem_required = activation_mem_required_gc
        print0(
            f"Using gradient checkpointing as intrucsted by user, activation memory required: {activation_mem_required} GB"
        )

    args.zero_stage = int(
        args.zero_stage) if args.zero_stage.isnumeric() else args.zero_stage
    assert args.zero_stage in [0, 1, 2, 3, "auto"
                               ], f"Invalid ZeRO stage: {args.zero_stage}"

    if args.zero_stage == 0:
        print0(
            f"Total per-GPU memory required w. current config: {z0_model_states_mem_required + activation_mem_required}"
        )
        if z0_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(
                f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                f"exceeds GPU memory ({z0_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB)."
            )
            print0(
                f"Consider using gradient_checkpointing, ZeRO-1, ZeRO-2, or ZeRO-3."
            )
            exit()
    elif args.zero_stage == 1:
        print0(
            f"Total per-GPU memory required w. current config: {z1_model_states_mem_required + activation_mem_required}"
        )
        if z1_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(
                f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                f"exceeds GPU memory ({z1_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB)."
            )
            print0(f"Consider using gradient_checkpointing, ZeRO-2 or ZeRO-3.")
            exit()
    elif args.zero_stage == 2:
        print0(
            f"Total per-GPU memory required w. current config: {z2_model_states_mem_required + activation_mem_required}"
        )
        if z2_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(
                f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                f"exceeds GPU memory ({z2_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB)."
            )
            print0(f"Consider using gradient_checkpointing, ZeRO-3.")
            exit()
    elif args.zero_stage == 3:
        print0(
            f"Total per-GPU memory required w. current config: {z3_model_states_mem_required + activation_mem_required}"
        )
        if z3_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(
                f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                f"exceeds GPU memory ({z3_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB)."
            )
            print0(
                f"ZeRO-1/2/3 are not suffecient, consider using more GPUs or a smaller model if possible."
            )
            exit()

    if args.zero_stage == "auto":
        if z0_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 0
        elif z0_model_states_mem_required + activation_mem_required_gc < mem_per_gpu:
            args.zero_stage = 0
            args.gradient_checkpointing = True
        elif z1_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 1
        elif z1_model_states_mem_required + activation_mem_required_gc < mem_per_gpu:
            args.zero_stage = 1
            args.gradient_checkpointing = True
        elif z2_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 2
        elif z2_model_states_mem_required + activation_mem_required_gc < mem_per_gpu:
            args.zero_stage = 2
            args.gradient_checkpointing = True
        elif z3_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 3
        elif z3_model_states_mem_required + activation_mem_required_gc < mem_per_gpu:
            args.zero_stage = 3
            args.gradient_checkpointing = True
        else:
            raise RuntimeError(
                f"Unable to fit model states + activation memory into GPU memory ({mem_per_gpu:.2f} GB). "
            )
        print0(f"Auto-selecting ZeRO stage: {args.zero_stage}" +
               (f" with gradient checkpointing" if args.
                gradient_checkpointing else " without gradient checkpointing"))

    print0(f"Using ZeRO stage: {args.zero_stage}" +
           (f" with gradient checkpointing" if args.
            gradient_checkpointing else " without gradient checkpointing"))
    return args


def _create_meta_model(model_name_or_path, model_class):
    model_config = AutoConfig.from_pretrained(model_name_or_path)

    with OnDevice(dtype=torch.float16, device='meta'):
        model = model_class.from_config(model_config)

    return model, model_config


def _apply_lora(meta_model, args, lora_dim, lora_module_name):
    meta_model = convert_linear_layer_to_lora(meta_model, lora_module_name,
                                              lora_dim)
    if args.only_optimize_lora:
        meta_model = only_optimize_lora_parameters(meta_model)
    return meta_model


def _activation_memory_estimate(meta_config, lora_dim, gradient_ckpt, seq_len,
                                batch_size):
    layers = meta_config.num_hidden_layers
    hd = meta_config.hidden_size
    seq = seq_len
    batch = batch_size
    vocab = meta_config.vocab_size
    heads = meta_config.num_attention_heads

    scale = 1e9

    # =9*I18*I19*I20*I17*2/1000000000
    gemms = 9 * hd * seq * batch * layers * 2 / scale
    # print0(f"{gemms=} GB")

    # =2*I20*I19*I19*I22*I17*2/1000000000
    attn = 2 * batch * seq * seq * heads * layers * 2 / scale
    # print0(f"{attn=} GB")

    # =2*I19*I20*I18*I17*2/1000000000
    ln = 2 * seq * batch * hd * layers * 2 / scale
    # print0(f"{ln=} GB")

    # =4*I18*I20*I19*I17*2/1000000000
    gelu = 4 * hd * batch * seq * layers * 2 / scale
    # print0(f"{gelu=} GB")

    # =2 *I20*I19*I21*2/1000000000
    loss = 2 * batch * seq * vocab * 2 / scale
    # print0(f"{loss=} GB")
    # total = gemms + attn + ln + gelu + loss + lora_activations

    lora_activations = 0
    if lora_dim > 0:
        # num_matrix = 4 # qkv fused (eg. bloom)
        num_matrix = 6  # qkv unfused (eg. opt)
        lora_activations = (seq * batch * lora_dim * layers * num_matrix *
                            2) / scale
        lora_activations += gemms
    print(f"{lora_activations=} GB")

    if gradient_ckpt:
        act_mem = (seq * batch * hd * 2 * layers) / scale
    else:
        act_mem = seq * batch * hd * layers * (34 + 5 * ((heads * seq) / hd))
        act_mem /= scale

    return act_mem + lora_activations
