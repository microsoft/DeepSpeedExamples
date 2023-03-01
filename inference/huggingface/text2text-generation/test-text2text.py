from argparse import ArgumentParser
import transformers
import deepspeed
import torch
import os
import time
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from deepspeed.runtime.utils import see_memory_usage

parser = ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="model_id")
parser.add_argument("--ds_inference", action='store_true', help="enable ds-inference")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
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
                inputs=["test"],
                num_tokens=50
                ):
        outputs = self.generate_outputs(inputs,num_tokens=num_tokens)
        return outputs

    def generate_outputs(self,
                         inputs=["test"],
                        num_tokens=50,
                        ):
        #generate_kwargs = dict(max_new_tokens=num_tokens)
        #input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        # for t in input_tokens:
        #     if torch.is_tensor(input_tokens[t]):
        #         input_tokens[t] = input_tokens[t].to(self.device)
        # self.model.cuda().to(self.device)
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        self.model.cuda().to(self.device)
        outputs = self.model.generate(inputs["input_ids"].to(self.device), max_new_tokens=args.max_new_tokens)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

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
        if args.dtype == "float16":
            num_bytes = 2
        elif args.dtype == "float32":
            num_bytes = 4
        else:
            num_bytes = 1
        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * args.batch_size / 1e12))


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
         "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy",
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
    outputs = pipe(inputs,num_tokens=args.max_new_tokens)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
    if args.test_performance:
        print_perf_stats(map(lambda t: t / args.max_new_tokens, times), pipe.model.config)


# pipe.device = torch.device(f'cuda:{local_rank}')
# output = pipe("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy",num_tokens=args.max_new_tokens)

# if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
#     print(output)
