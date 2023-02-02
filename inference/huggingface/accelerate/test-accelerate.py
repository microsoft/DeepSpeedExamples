import os
import math
import time
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from argparse import ArgumentParser
from deepspeed.runtime.utils import see_memory_usage

parser = ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
args = parser.parse_args()

def print_perf_stats(latency_set, config, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        num_bytes = 2
        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * args.batch_size / 1e12))

checkpoint = "EleutherAI/gpt-j-6B"
weights_path = snapshot_download(checkpoint)
files = os.listdir(weights_path)
weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = load_checkpoint_and_dispatch(
    model, weights_path, device_map="auto", no_split_module_classes=["GPTJBlock"]
)

print(model.hf_device_map)
see_memory_usage("after load_checkpoint_and_dispatch", True)

input_sentences = [
         "He is working on",
         "DeepSpeed is a machine learning framework",
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

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

iters=30
times=[]
for i in range(iters):
    torch.cuda.synchronize()
    start = time.time()
    output = model.generate(inputs["input_ids"].to(0),
                            attention_mask=inputs['attention_mask'].to(0),
                            max_new_tokens=50
                            )
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

outputs = tokenizer.batch_decode(output, skip_special_tokens=True)

for output in outputs:
   print("------------------------------------------------------------")
   print(output)

print_perf_stats(map(lambda t: t / 50, times), model.config)
