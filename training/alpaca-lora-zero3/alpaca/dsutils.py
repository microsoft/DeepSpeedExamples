import os
from typing import Optional, Dict, TypedDict

import hjson
import megfile
from tqdm import tqdm
from loguru import logger

import torch
import torch.distributed
import deepspeed
from torch import nn
from deepspeed import comm as dist
from deepspeed.utils import safe_get_full_fp32_param
from deepspeed.runtime.zero.partition_parameters import is_zero_param
from peft import PeftModel


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


class _ParamStat(TypedDict):
    trainable: int
    total: int


def trainable_parameters_stat(model) -> _ParamStat:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += (param.ds_numel if is_zero_param(param) else param.numel())
        if param.requires_grad:
            trainable_params += (param.ds_numel if is_zero_param(param) else param.numel())

    return {
        "trainable": trainable_params,
        "total": all_param,
    }


def zero3_load_pretrained(model: nn.Module, pretrained_dir: megfile.SmartPath):
    pretrained_dir = megfile.SmartPath(pretrained_dir)
    assert pretrained_dir.is_dir()

    is_rank_0 = torch.distributed.get_rank() == 0

    def _load_checkpoint_file(ckp_file):
        if is_rank_0:
            # Load state dict on rank-0
            with (ckp_file).open("rb") as f:
                state_dict = torch.load(f, map_location="cpu")
            meta_state_dict = [{k: None for k in state_dict}]
        else:
            meta_state_dict = [None, ]
        # broadcast param names
        torch.distributed.broadcast_object_list(meta_state_dict, src=0)
        if not is_rank_0:
            state_dict = meta_state_dict[0]
        dist.barrier()
        return zero3_load_state_dict_into_model(model, state_dict, "")

    index_file = pretrained_dir / "pytorch_model.bin.index.json"
    # single checkpoint
    if not index_file.is_file():
        ckp_file = pretrained_dir / "pytorch_model.bin"
        logger.info(f"Loading {ckp_file}")
        return _load_checkpoint_file(ckp_file)

    # Multiple shared checkpoint
    err_msgs = []
    with index_file.open() as f:
        index = hjson.load(f)
    shard_files = tqdm(sorted(set(index['weight_map'].values())), dynamic_ncols=True, unit="file", disable=(not is_rank_0))
    for shard_file in shard_files:
        shard_files.set_description(shard_file)
        ckp_file = pretrained_dir / shard_file
        err_msgs += _load_checkpoint_file(ckp_file)

    return err_msgs


def zero3_load_state_dict_into_model(model_to_load: nn.Module, state_dict: Dict[str, Optional[torch.Tensor]], start_prefix: str = ''):
    """
    Modified from huggingface `transformers.modeling_utils._load_state_dict_into_model`

    On rank-0, state_dict has tensors; on other ranks, state_dict only contains parameter names (In order to synchronize `GateredParams`)
    """
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)
            else:
                if torch.distributed.get_rank() == 0:
                    module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


def zero3_get_peft_model_state_dict(model: PeftModel, adapter_name="default"):
    from peft.utils import PeftType

    config = model.peft_config[adapter_name]
    if config.peft_type != PeftType.LORA:
        raise Exception("Only LoRA is supported")

    all_params = set(k for k, _ in model.named_parameters())
    bias = config.bias
    if bias == "none":
        candidate_keys = set(k for k in all_params if "lora_" in k)
    elif bias == "all":
        candidate_keys = set(k for k in all_params if "lora_" in k or "bias" in k)
    elif bias == "lora_only":
        candidate_keys = set(k for k in all_params if "lora_" in k)
        for k in candidate_keys:
            bias_name = k.split("lora_")[0] + "bias"
            if bias_name in all_params:
                candidate_keys.add(bias_name)

    candidate_keys = set(k for k in candidate_keys if (("lora_" in k and adapter_name in k) or ("bias" in k)))

    state_dict = {
        k: safe_get_full_fp32_param(v).cpu() for k, v in model.named_parameters()
        if k in candidate_keys
    }
    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in state_dict.items()}
    return to_return


def zero3_save_lora_model(model: PeftModel, save_directory: str):
    """Modified from PeftModel.save_pretrained"""
    WEIGHTS_NAME = "adapter_model.bin"

    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

    os.makedirs(save_directory, exist_ok=True)

    for adapter_name, peft_config in model.peft_config.items():
        # peft 0.3 has multi-adapter support, so we need to save each adapter separately
        # save only the trainable weights
        output_state_dict = zero3_get_peft_model_state_dict(model, adapter_name=adapter_name)
        output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
        os.makedirs(output_dir, exist_ok=True)
        torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = getattr(model.base_model, "name_or_path", None)
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True
        peft_config.save_pretrained(output_dir)
        peft_config.inference_mode = inference_mode

    logger.info(f"LoRA weights saved to {save_directory}")