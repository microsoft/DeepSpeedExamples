from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
from utils import DSPipeline


def bool_arg(x):
    return x.lower() =='true'

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--ds_inference", default=True, type=lambda x : bool_arg(x), help="enable ds-inference")
parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--greedy", default=False, type=lambda x : bool_arg(x), help="greedy generation mode")
parser.add_argument("--use_meta_tensor", default=False, type=lambda x : bool_arg(x), help="use the meta tensors to initialize model")
parser.add_argument("--hf_low_cpu_mem_usage", default=False, type=lambda x : bool_arg(x), help="use the low_cpu_mem_usage flag in huggingface to initialize model")
parser.add_argument("--use_cache", default=True, type=lambda x : bool_arg(x), help="use cache for generation")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

world_size = int(os.getenv('WORLD_SIZE', '1'))

if args.use_meta_tensor and args.hf_low_cpu_mem_usage:
    raise ValueError("Cannot use both use_meta_tensor and hf_low_cpu_mem_usage")

data_type = getattr(torch, args.dtype)
pipe = DSPipeline(model_name=args.name,
                  dtype=data_type,
                  is_meta=args.use_meta_tensor,
                  is_hf_low_cpu_mem_usage=args.hf_low_cpu_mem_usage,
                  device=args.local_rank)

if args.use_meta_tensor:
    ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
else:
    ds_kwargs = dict()

if args.ds_inference:
    pipe.model = deepspeed.init_inference(pipe.model,
                                    dtype=data_type,
                                    mp_size=world_size,
                                    replace_with_kernel_inject=True,
                                    max_tokens=args.max_tokens,
                                    **ds_kwargs
                                    )

input_sentences = [
         "DeepSpeed is a machine learning framework",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way"
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

outputs = pipe(inputs,
              num_tokens=args.max_new_tokens,
              do_sample=(not args.greedy))

for i, o in zip(inputs, outputs):
    print(f"\nin={i}\nout={o}\n{'-'*60}")

