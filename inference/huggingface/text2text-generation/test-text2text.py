from argparse import ArgumentParser
import transformers
import deepspeed
import torch
import os
import time
import math
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from deepspeed.runtime.utils import see_memory_usage

parser = ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="model_id")
parser.add_argument("--ds_inference", action='store_true', help="enable ds-inference")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--test_performance", action='store_true', help="enable latency, bandwidth, and throughout testing")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

class DSPipeline():
    def __init__(self,
                 model_name='t5-11b',
                 dtype=torch.float16,
                 device=-1,
                 checkpoint_path=None
                 ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", model_max_length=512)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()

    def __call__(self,
                inputs=["test"]
                ):
        outputs = self.generate_outputs(inputs)
        return outputs

    def generate_outputs(self,
                         inputs=["test"]
                        ):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        self.model.cuda().to(self.device)
        outputs = self.model.generate(inputs["input_ids"].to(self.device))
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        average_num_tokens = 0
        for o in outputs:
            average_num_tokens += len(self.tokenizer.tokenize(o))
        average_num_tokens = average_num_tokens/args.batch_size
        return outputs, average_num_tokens

def print_perf_stats(latency_set, config, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = 11 * 1e9 #num_layers * config.hidden_size * config.hidden_size * 12
        if args.dtype == "float16":
            num_bytes = 2
        elif args.dtype == "float32":
            num_bytes = 4
        else:
            num_bytes = 1

        log = open("log.txt","a")
        log.write(str(os.getenv('WORLD_SIZE', '1')) + " gpus, " + str(args.batch_size) + " batch\n")
        log.write(str(num_parameters))
        log.write(str(num_bytes))
        log.write("Avg Per Token Latency: {0:8.2f} ms\n".format(avg * 1000))
        log.write("Avg BW: {0:8.2f} GB/s\n".format(1/avg * num_parameters * num_bytes / 1e9))
        log.write("Avg flops: {0:8.2f} TFlops/s\n".format(1/avg * num_parameters * num_bytes / 1e12))
        log.close()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

data_type = getattr(torch, args.dtype)

pipe = DSPipeline(model_name=args.model,
                  dtype=data_type,
                  device=args.local_rank,
                  )

if local_rank == 0:
    see_memory_usage("before init", True)

if args.ds_inference:
    pipe.model = deepspeed.init_inference(
        pipe.model,
        mp_size=world_size,
        dtype=data_type,
    )

if local_rank == 0:
    see_memory_usage("after init", True)

input_sentences = [
         "DeepSpeed is a machine learning framework",
         "summarize: My friends are cool but they eat too many carbs",
         "summarize: There are many reasons to have a dog",
         "translate English to French: He is working on it",
         "summarize: My friends are cool but they eat too many carbs.",
         "translate English to German: The house is wonderful.",
         "summarize: studies have shown that owning a dog is good for you",
         "translate English to Spanish: The new movie that got Oscar this year",
         "translate English to French: In the far far distance from our galaxy,",
         "translate English to German: Peace is the only way."
]

if args.batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(args.batch_size / len(input_sentences))

inputs = input_sentences[:args.batch_size]

iters = 30 if args.test_performance else 1
times=[]
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    outputs, average_num_tokens = pipe(inputs)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    # for i, o in zip(inputs, outputs):
    #     print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        print_perf_stats(map(lambda t: t / (average_num_tokens*args.batch_size), times), pipe.model.config)
