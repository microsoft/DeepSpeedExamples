from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
import math
import os
import torch
import time
from utils import DSPipeline
from transformers.models.opt.modeling_opt import OPTDecoderLayer

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--replace_method", required=False, default='', type=str, help="replace method['', 'auto', 'dict']")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--ds_inference", action='store_true', help="enable ds-inference")
parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--greedy", default=False, type=bool, help="greedy generation mode")
parser.add_argument("--use_meta_tensor", action='store_true', help="use the meta tensors to initialize model")
parser.add_argument("--use_cache", default=True, type=bool, help="use cache for generation")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

def print_latency(latency_set, title, config, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)
#    print(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print(f"====== latency stats {title} ======")
        print("\tAvg Latency: {0:8.2f} ms".format(avg * 1000))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 * 1000))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 * 1000))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 * 1000))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 * 1000))
        print("\t999 Latency: {0:8.2f} ms".format(p999 * 1000))
        num_layers = config.num_layers if hasattr(config,'num_layers') else config.num_hidden_layers
        print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_layers * config.hidden_size * config.hidden_size * 12 * 2 / 1000000000))
        print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_layers * config.hidden_size * config.hidden_size * 12 * 2 / 1000000000000 * args.batch_size))


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))
print(world_size)
data_type = getattr(torch, args.dtype)
pipe = DSPipeline(model_name=args.name,
                  dtype=data_type,
                  is_meta=args.use_meta_tensor,
                  device=args.local_rank)

if args.use_meta_tensor:
    ds_kwargs = dict(base_dir=pipe.repo_root, checkpoint=pipe.checkpoints_json)
else:
    ds_kwargs = dict()

if args.ds_inference:
    pipe.model = deepspeed.init_inference(pipe.model,
                                    dtype=data_type,
                                    mp_size=world_size,
                                    replace_with_kernel_inject=args.use_kernel,
                                    replace_method=args.replace_method,
#                                    injection_policy={
#                                        OPTDecoderLayer: ()
#                                    },
                                    max_tokens=args.max_tokens,
                                    **ds_kwargs
                                    )

    

responses = []
times = []
mtimes = []

#print(pipe.model)

input_sentences = [
         "DeepSpeed is a machine learning framework",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way",
         "She went to the",
         "This little piggy went to the", #10
         "Have you heard that",
         "The greatest artist of all time",
         "Once upon a time there was",
         "A moose, a fish, and a bird walk into a bar",
         "She was looking for",
         "DeepSpeed will solve",
         "The president will be",
         "The weather is",
         "Tomorrow will be the first time",
         "Yesterday was", #20
         "I will be going to",
         "Her friends all thought that",
         "In order to make a pie you",
         "The recipe for cake is",
         "During the apocalypse",
         "Everyone wants",
         "This holiday season will be",
         "Unlike other types of fish",
         "Fish think about",
         "Fish are friends not", #30
         "DeepSpeed is a machine learning framework",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way",
         "She went to the",
         "This little piggy went to the", #10
         "Have you heard that",
         "The greatest artist of all time",
         "Once upon a time there was",
         "A moose, a fish, and a bird walk into a bar",
         "She was looking for",
         "DeepSpeed will solve",
         "The president will be",
         "The weather is",
         "Tomorrow will be the first time",
         "Yesterday was", #20
         "I will be going to",
         "Her friends all thought that",
         "In order to make a pie you",
         "The recipe for cake is",
         "During the apocalypse",
         "Everyone wants",
         "This holiday season will be",
         "Unlike other types of fish",
         "Fish think about",
         "Fish are friends not", #30
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

for i in range(30):
    torch.cuda.synchronize()
    start = time.time()

    outputs = pipe(inputs,
              num_tokens=args.max_new_tokens,
              do_sample=(not args.greedy))

    torch.cuda.synchronize()
    end = time.time()

    times.append(end - start)  # / (args.max_tokens - 3))


#for i, o in zip(inputs, outputs):
#    print(f"\nin={i}\nout={o}\n{'-'*60}")

if args.local_rank == 0:
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")

    print_latency(map(lambda t: t / (args.max_new_tokens),
                      times),
                  "(e2e) per token latency", pipe.model.config)

