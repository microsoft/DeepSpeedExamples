# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


def print_throughput(hf_model,
                     args,
                     e2e_time,
                     gen_exp_time,
                     gen_seq_time,
                     train_time,
                     rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers = getattr(hf_config, "num_hidden_layers",
                             getattr(hf_config, "n_layer", None))
        hidden_size = getattr(hf_config, "hidden_size",
                              getattr(hf_config, "n_embd", None))
        vocab_size = getattr(hf_config, "vocab_size", None)
        assert all(
            (num_layers, hidden_size, vocab_size)
        ), "Could not determine number of layers, hidden size, and vocab size of the model"

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_answer_seq_len
        batch_size = args.per_device_training_batch_size * gpus_per_model
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.actor_gradient_checkpointing else 3
        hf_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in hf_model.parameters()
        ])
        params_in_billions = hf_model._num_params / (1e9)

        # megatron paper formula
        flops_per_iteration = (24 * checkpoint_activations_factor *
                               batch_size * seq_length * num_layers *
                               (hidden_size**2)) * (
                                   1.0 + (seq_length / (6.0 * hidden_size)) +
                                   (vocab_size /
                                    (16.0 * num_layers * hidden_size)))
        e2e_tflops = flops_per_iteration / (e2e_time * gpus_per_model *
                                            (10**12))
        gen_tflops = flops_per_iteration / (gen_exp_time * gpus_per_model *
                                            (10**12))
        train_tflops = flops_per_iteration / (train_time * gpus_per_model *
                                              (10**12))
        print(
            f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {e2e_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )
        print(
            f"Generation => Latency: {gen_exp_time:.2f}s, TFLOPs: {gen_tflops:.2f}"
        )
        print(
            f"Training   => Latency: {train_time:.2f}s, TFLOPs: {train_tflops:.2f}"
        )
        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(f"Parameters => {param_string}")
