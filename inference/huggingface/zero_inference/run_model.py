"""
Run OPT with huggingface or deepspeed.

Reference:
https://github.com/FMInference/FlexGen/blob/main/benchmark/hf_ds/hf_opt.py
"""

import argparse
import gc
import multiprocessing as mp
import os
import pickle

import deepspeed
import torch
import torch.distributed as dist
from accelerate import (infer_auto_device_map, init_empty_weights,
                        load_checkpoint_and_dispatch)
from timer import timers
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BloomForCausalLM, OPTForCausalLM)
from transformers.deepspeed import HfDeepSpeedConfig
from utils import (GB, add_model_hooks, cache_bytes, disable_torch_init,
                   get_filename, get_quant_config, hidden_bytes, meta_to_cpu,
                   model_bytes, write_benchmark_log)
from packaging import version


assert version.parse(deepspeed.__version__) >= version.parse("0.10.1"), "ZeRO-Inference v2.0 is available only in DeepSpeed 0.10.1+, please upgrade DeepSpeed"


def get_model_config(model_name):
    if "175b" in model_name:
        config = AutoConfig.from_pretrained("facebook/opt-66b")
        config.hidden_size = 12288
        config.word_embed_proj_dim = 12288
        config.ffn_dim = 12288 * 4
        config.num_attention_heads = 96
        config.num_hidden_layers = 96
    else:
        config = AutoConfig.from_pretrained(model_name)

    if 'bloom' in model_name:
        config.model_type = 'bloom'

    return config

def get_ds_model(
    model_name,
    dtype,
    cpu_offload,
    disk_offload,
    offload_dir,
    dummy_weights,
    bits,
    group_size,
):

    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()
    pin_memory = bool(args.pin_memory)

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 0,  # 2 * hidden_size * hidden_size,
            "stage3_param_persistence_threshold": hidden_size,
            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": args.batch_size,
        "wall_clock_breakdown": False,
    }

    if bits == 4:
        quant_config = get_quant_config(config, bits=bits, group_size=group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )

    if disk_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="nvme",
            pin_memory=pin_memory,
            nvme_path=offload_dir,
            buffer_count=5,
            buffer_size=9 * GB if config.model_type == 'bloom' else 2 * GB,
        )
        ds_config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True,
        }

    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    torch.cuda.empty_cache()
    gc.collect()

    if config.model_type in ["bloom", "bloom-7b1"]:
        model = BloomForCausalLM.from_pretrained(
            dummy_weights or model_name, torch_dtype=dtype
        )
    elif config.model_type == "opt":
        model = OPTForCausalLM.from_pretrained(
            dummy_weights or model_name, torch_dtype=dtype
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    model = model.eval()


    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(f"model.config = {model.config}")

    return model


def get_hf_model(
    model_name, dtype, cpu_offload, disk_offload, offload_dir, num_gpus, dummy_weights
):
    if num_gpus == 1 and dtype != torch.int8:
        # Here we use a custom device_map instead of device_map == "auto"
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        if cpu_offload:
            # NOTE: We must put some weights on GPU. Otherwise, huggingface reports errors.
            device_map = {
                "model.decoder.embed_tokens.weight": 0,
                "model.decoder.embed_positions.weight": 0,
                "model.decoder.final_layer_norm": "cpu",
                "model.decoder.layers": "cpu",
                "lm_head.weight": 0,
            }
        elif disk_offload:
            device_map = {
                "model.decoder.embed_tokens.weight": 0,
                "model.decoder.embed_positions.weight": 0,
                "model.decoder.final_layer_norm": "disk",
                "model.decoder.layers": "disk",
                "lm_head.weight": 0,
            }
        else:
            device_map = None
        max_memory = None
    else:
        # Here we use device_map == "auto", but set a low `max_memory` threshold
        # becase we want to offload as many as possible weights out of GPU
        # to allow a larger batch size.
        device_map = "auto"
        if cpu_offload:
            # `max_memory` should be larger than the embedding.
            # We use 2GB here because the embeding of opt-175b is 1.2GB.
            max_memory = {k: "2GB" for k in range(num_gpus)}
        elif disk_offload:
            max_memory = {k: "2GB" for k in range(num_gpus)}
        else:
            max_memory = {k: "14GB" for k in range(num_gpus)}
        max_memory["cpu"] = "160GB"

    if dtype == torch.int8:
        kwargs = {"load_in_8bit": True}
    else:
        kwargs = {"torch_dtype": dtype}

    disable_torch_init()
    model = OPTForCausalLM.from_pretrained(
        dummy_weights or model_name,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_dir,
        **kwargs,
    )
    if device_map is None:
        model.cuda()

    model.eval()
    return model


def run_generation(
    model_name,
    batch_size,
    prompt_len,
    gen_len,
    cpu_offload,
    disk_offload,
    offload_dir,
    use_int8,
    num_nodes,
    num_gpus_per_node,
    use_deepspeed,
    dummy,
    output_file,
    verbose,
    kv_offload,
    quant_bits,
    quant_group_size,
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name.replace("175b", "66b"), padding_side="left"
    )

    # Load model
    if use_int8:
        dtype = torch.int8
    else:
        dtype = torch.float16

    config = get_model_config(model_name)

    if dummy:
        filename = os.path.join(
            offload_dir, f"{model_name.replace('/', '-')}-hf-weights/"
        )
        if not os.path.exists(filename):
            print("create dummy weights")
            with init_empty_weights():
                if 'opt' in model_name:
                    model = OPTForCausalLM(config)
                else:
                    model = BloomForCausalLM(config)
            model.save_pretrained(
                filename, state_dict=meta_to_cpu(model.state_dict(), torch.float16)
            )
        dummy_weights = filename
    else:
        dummy_weights = None

    print("load model")
    if use_deepspeed:
        with torch.no_grad():
            model = get_ds_model(
                model_name,
                dtype,
                cpu_offload,
                disk_offload,
                offload_dir,
                dummy_weights,
                quant_bits,
                quant_group_size,
            )
    else:
        model = get_hf_model(
            model_name,
            dtype,
            cpu_offload,
            disk_offload,
            offload_dir,
            num_gpus_per_node,
            dummy_weights,
        )

    # Run generation
    execute_gen_len = gen_len
    if use_deepspeed:
        prompts = ["Paris is the capital city of"] * (batch_size // WORLD_SIZE)
    else:
        prompts = ["Paris is the capital city of"] * batch_size
    input_ids = tokenizer(
        prompts, return_tensors="pt", padding="max_length", max_length=prompt_len
    ).input_ids.cuda()

    if kv_offload:
        # set kv_offload in model config
        model.config.kv_offload = True
        model.config.max_new_tokens = gen_len

    print(model, model.config)

    add_model_hooks(model)

    def set_model_stage(model, stage):
        model.stage = stage

    # Run
    print(f"benchmark, execute_gen_len = {execute_gen_len}, input_ids.shape = {input_ids.shape}")
    generate_kwargs = dict(max_new_tokens=execute_gen_len, do_sample=False)
    prefill_timings = []
    timer = timers("generate-forward")
    for _ in range(3):
        timer.start(sync_func=torch.cuda.synchronize)
        with torch.no_grad():
            set_model_stage(model, "prefill")
            output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
            prefill_timings.append(model.__duration__)
        timer.stop(sync_func=torch.cuda.synchronize)
    costs = timers("generate-forward").costs

    if use_deepspeed and args.local_rank != 0:
        return

    def remove_model_hooks(module):
        if hasattr(module, "__start_time_hook_handle__"):
            module.__start_time_hook_handle__.remove()
            del module.__start_time_hook_handle__
        if hasattr(module, "__end_time_hook_handle__"):
            module.__end_time_hook_handle__.remove()
            del module.__end_time_hook_handle__
        if hasattr(module, "stage"):
            del module.stage
        if hasattr(module, "__duration__"):
            del module.__duration__

    # Log output
    print(f"Summary:")
    print(f"costs = {costs}, prefill_timings = {prefill_timings}")
    total_latency = costs[-1]
    prefill_latency = prefill_timings[-1]
    remove_model_hooks(model)

    prefill_throughput = batch_size * prompt_len / prefill_latency
    decode_latency = total_latency - prefill_latency
    decode_throughput = batch_size * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = batch_size * gen_len
    total_throughput = num_generated_tokens / total_latency
    gpu_peak_mem = torch.cuda.max_memory_allocated(torch.device("cuda"))
    out_str = ""

    if verbose >= 2:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * "-" + "\n"
        for i in [0, (len(outputs) - 1) // 2, len(outputs) - 1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += 70 * "-" + "\n"
        print(show_str)

        # Check lengths
        input_lens = [len(x) for x in input_ids]
        output_lens = [len(x) for x in output_ids]
        assert all(x == prompt_len for x in input_lens)
        assert all(x == prompt_len + execute_gen_len for x in output_lens)

    if args.log_file == "auto":
        filename = (
            get_filename(
                model_name,
                batch_size,
                prompt_len,
                gen_len,
                cpu_offload,
                disk_offload,
                num_nodes,
                num_gpus_per_node,
                use_deepspeed,
            )
            + ".log"
        )
    else:
        filename = args.log_file

    cache_size = cache_bytes(config, batch_size, prompt_len + gen_len)
    hidden_size = hidden_bytes(config, batch_size, prompt_len + gen_len)
    log_str = write_benchmark_log(
        filename,
        model_bytes(config),
        cache_size,
        hidden_size,
        gpu_peak_mem,
        prefill_latency,
        prefill_throughput,
        decode_latency,
        decode_throughput,
        total_latency,
        total_throughput,
    )
    if verbose >= 1:
        print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="model name or path; currently only supports OPT and BLOOM models")
    parser.add_argument(
        "--dummy", action="store_true", help="Use dummy weights for benchmark purposes."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512,  help="prompt length")
    parser.add_argument("--gen-len", type=int, default=32,  help="number of tokens to generate")
    parser.add_argument("--local_rank", type=int, help="local rank for distributed inference")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--pin-memory", type=int, default=0, help="whether to pinned CPU memory for ZeRO offloading")
    parser.add_argument("--cpu-offload", action="store_true", help="Use cpu offload.")
    parser.add_argument("--disk-offload", action="store_true", help="Use disk offload.")
    parser.add_argument("--offload-dir", type=str, default="~/offload_dir", help="Directory to store offloaded cache.")
    parser.add_argument(
        "--kv-offload", action="store_true", help="Use kv cache cpu offloading."
    )
    parser.add_argument("--int8", action="store_true", help="Use HuggingFace int8 quantization.")

    parser.add_argument("--log-file", type=str, default="auto", help="log file name")
    parser.add_argument("--verbose", type=int, default=2, help="verbose level")
    parser.add_argument("--quant_bits", type=int, default=16, help="model weight quantization bits; either 4 or 8")
    parser.add_argument("--quant_group_size", type=int, default=64, help="model weight quantization group size")
    args = parser.parse_args()

    if args.local_rank is None:  # huggingface accelerate
        use_deepspeed = False
        num_gpus_per_node = args.num_gpus
        num_nodes = 1
    else:  # deepspeed
        use_deepspeed = True
        WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
        num_gpus_per_node = torch.cuda.device_count()
        num_nodes = WORLD_SIZE // num_gpus_per_node

    print(f"use_deepspeed = {use_deepspeed}")

    run_generation(
        args.model,
        args.batch_size,
        args.prompt_len,
        args.gen_len,
        args.cpu_offload,
        args.disk_offload,
        os.path.abspath(os.path.expanduser(args.offload_dir)),
        args.int8,
        num_nodes,
        num_gpus_per_node,
        use_deepspeed,
        args.dummy,
        args.log_file,
        args.verbose,
        args.kv_offload,
        args.quant_bits,
        args.quant_group_size,
    )
