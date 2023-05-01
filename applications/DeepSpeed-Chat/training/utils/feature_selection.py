import torch

from deepspeed.utils import OnDevice
from transformers import AutoConfig

from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters


def print0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)    


def feature_selection_step3(args, actor, critic):
    
    # create meta actor (trainable)

    if args.lora_dim > 0:
        # apply lora to meta actor

    # create meta critic (trainable)

    # create meta reference model based on actor

    if args.enable_ema:
        # create meta ema model based on actor
    
    # create meta reward model based on critic


def feature_selection(args, model_class):
    meta_model, model_config = _create_meta_model(args.model_name_or_path, model_class)
    nparams = sum([p.numel() for p in meta_model.parameters()])
    print0(f"[pre-lora] num params: {nparams}")

    if args.lora_dim > 0:
        meta_model = _apply_lora(meta_model, args)

    nparams = sum([p.numel() for p in meta_model.parameters()])
    print0(f"[post-lora] num params: {nparams}")

    # [pre-LoRA] num params: 1,315,758,080
    # [post-LoRA] num params: 1,429,004,288
    # LoRA adds 113,246,208 parameters

    trainable_params = sum([p.numel() if p.requires_grad else 0 for p in meta_model.parameters()])
    frozen_params = sum([p.numel() if not p.requires_grad else 0 for p in meta_model.parameters()])
    print0(f"{trainable_params=}, {frozen_params=}")

    #[pre-lora] num params: 1315758080
    #[post-lora] num params: 1429004288
    #trainable_params=221,044,736
    #frozen_params=1,207,959,552

    GB = 1024 ** 3
    world_size = torch.distributed.get_world_size()
    mem_per_gpu = torch.cuda.get_device_properties(0).total_memory / GB

    # model weights (fp16) + gradients (fp16) + optimizer states (fp16/fp32)
    z0_model_states_mem_required = (nparams * 2 + trainable_params * 2 + trainable_params * 12) / GB
    print0(f'[ZeRO=0] Total model/optim states required: {z0_model_states_mem_required} GB')

    z1_model_states_mem_required = nparams * 2 # model weights
    z1_model_states_mem_required += trainable_params * 2 # model grads
    z1_model_states_mem_required += (trainable_params * (12 + 4)) / world_size # sharded optim states + fp32 sharded grads
    z1_model_states_mem_required /= GB
    print0(f'[ZeRO=1] Total model/optim states required: {z1_model_states_mem_required} GB')

    z2_model_states_mem_required = nparams * 2 # model weights
    z2_model_states_mem_required += (trainable_params * 2) / world_size # model grads are sharded
    z2_model_states_mem_required += (trainable_params * (12 + 4)) / world_size # sharded optim states + fp32 sharded grads
    z2_model_states_mem_required /= GB
    print0(f'[ZeRO=2] Total model/optim states required: {z2_model_states_mem_required} GB')

    z3_model_states_mem_required = (nparams * 2) / world_size # model weights are sharded
    z3_model_states_mem_required += (trainable_params * 2) / world_size # model grads are sharded
    z3_model_states_mem_required += (trainable_params * (12 + 4)) / world_size # sharded optim states + fp32 sharded grads
    z3_model_states_mem_required /= GB
    print0(f'[ZeRO=3] Total model/optim states required: {z3_model_states_mem_required} GB')

    activation_mem_required = _activation_memory_estimate(model_config, args)
    print0(f"Estimated activation memory required: {activation_mem_required} GB")

    if args.zero_stage == 0:
        print0(f"Total per-GPU memory required w. current config: {z0_model_states_mem_required + activation_mem_required}")
        if z0_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                  f"exceeds GPU memory ({z0_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB).")
            print0(f"Consider using ZeRO-1, ZeRO-2, or ZeRO-3.")
            exit()
    elif args.zero_stage == 1:
        print0(f"Total per-GPU memory required w. current config: {z1_model_states_mem_required + activation_mem_required}")
        if z1_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                  f"exceeds GPU memory ({z1_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB).")
            print0(f"Consider using ZeRO-2 or ZeRO-3.")
            exit()
    elif args.zero_stage == 2:
        print0(f"Total per-GPU memory required w. current config: {z2_model_states_mem_required + activation_mem_required}")
        if z2_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                  f"exceeds GPU memory ({z2_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB).")
            print0(f"Consider using ZeRO-3.")
            exit()
    elif args.zero_stage == 3:
        print0(f"Total per-GPU memory required w. current config: {z3_model_states_mem_required + activation_mem_required}")
        if z3_model_states_mem_required + activation_mem_required > mem_per_gpu:
            print0(f"WARNING: Model states (model weights, gradients, optimizer states) + Activation memory "
                  f"exceeds GPU memory ({z3_model_states_mem_required:.2f} + {activation_mem_required:.2f} GB > {mem_per_gpu:.2f} GB).")
            print0(f"ZeRO-1/2/3 are not suffecient, consider using more GPUs or a smaller model if possible.")
            exit()

    #TODO(Cheng): auto-select if gradient checkpointing is enabled

    if args.zero_stage == "auto":
        if z0_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 0
        if z1_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 1
        elif z2_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 2
        elif z3_model_states_mem_required + activation_mem_required < mem_per_gpu:
            args.zero_stage = 3
        else:
            raise RuntimeError(f"Unable to fit model states + activation memory into GPU memory ({mem_per_gpu:.2f} GB). ")
        print0(f"Auto-selecting ZeRO stage: {args.zero_stage}")


def _create_meta_model(model_name_or_path, model_class):
    model_config = AutoConfig.from_pretrained(model_name_or_path)

    with OnDevice(dtype=torch.float16, device='meta'):
        model = model_class.from_config(model_config)

    return model, model_config


def _apply_lora(meta_model, args):
    meta_model = convert_linear_layer_to_lora(meta_model, args.lora_module_name,
                                            args.lora_dim)
    if args.only_optimize_lora:
        meta_model = only_optimize_lora_parameters(meta_model)
    return meta_model


def _activation_memory_estimate(model_config, args):
    layers = model_config.num_hidden_layers
    hd = model_config.hidden_size
    seq = args.max_seq_len
    batch = args.per_device_train_batch_size
    vocab = model_config.vocab_size
    heads = model_config.num_attention_heads

    scale = 1e9

    # =9*I18*I19*I20*I17*2/1000000000
    gemms = 9 * hd * seq * batch * layers * 2 / scale
    # print(f"{gemms=} GB")

    # =2*I20*I19*I19*I22*I17*2/1000000000
    attn = 2 * batch * seq * seq * heads * layers * 2 / scale
    # print(f"{attn=} GB")

    # =2*I19*I20*I18*I17*2/1000000000
    ln = 2 * seq * batch * hd * layers * 2 / scale
    # print(f"{ln=} GB")

    # =4*I18*I20*I19*I17*2/1000000000
    gelu = 4 * hd * batch * seq * layers * 2 / scale
    # print(f"{gelu=} GB")

    # =2 *I20*I19*I21*2/1000000000
    loss = 2 * batch * seq * vocab * 2 / scale
    # print(f"{loss=} GB")
    # total = gemms + attn + ln + gelu + loss + lora_activations

    lora_activations = 0
    if args.lora_dim > 0:
        # num_matrix = 4 # qkv fused (eg. bloom)
        num_matrix = 6 # qkv unfused (eg. opt)
        lora_activations = (seq * batch * args.lora_dim * layers * num_matrix * 2) / scale
        lora_activations += gemms
    print(f"{lora_activations=} GB")

    if args.gradient_checkpointing:
        act_mem = (seq * batch * hd * 2 * layers) / scale
    else:
        act_mem = seq * batch * hd * layers * (34 + 5 * ((heads * seq) / hd))
        act_mem /= scale

    return act_mem + lora_activations

