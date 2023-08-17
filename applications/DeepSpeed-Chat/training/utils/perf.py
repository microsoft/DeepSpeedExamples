# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import numpy as np
from loguru import logger
import torch

def print_throughput(hf_model, args, iter_times):
    hf_config = hf_model.config
    num_layers = getattr(
        hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", None)
    )
    hidden_size = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    gpus_per_model = torch.distributed.get_world_size()
    seq_length = args.max_answer_seq_len
    batch_size = args.per_device_mini_train_batch_size * gpus_per_model
    avg_iter_time = np.mean(iter_times)
    samples_per_second = batch_size / avg_iter_time
    checkpoint_activations_factor = 4 if args.actor_gradient_checkpointing else 3
    hf_model._num_params = sum(
            [
                p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
                for p in hf_model.parameters()
            ]
        )
    params_in_billions = hf_model._num_params / (1e9)

    flops_per_iteration = (
        24
        * checkpoint_activations_factor
        * batch_size
        * seq_length
        * num_layers
        * (hidden_size**2)
    ) * (
        1.0
        + (seq_length / (6.0 * hidden_size))
        + (vocab_size / (16.0 * num_layers * hidden_size))
    )
    tflops = flops_per_iteration / (avg_iter_time * gpus_per_model * (10**12))
    logger.info(
        ", ".join(
            (
                f"Time: {avg_iter_time:.2f}s",
                f"{samples_per_second:.2f} Samples/S",
                f"{tflops:.2f} TFLOPs",
                "Params: "
                + (f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"),
            )
        )
    )

    return {
        "Flops": tflops,
        "Time per step": avg_iter_time,
        "Billion params": params_in_billions,
    }