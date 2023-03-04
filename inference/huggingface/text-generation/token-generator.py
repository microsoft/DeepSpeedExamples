from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import deepspeed
import math
import os
import torch

parser = ArgumentParser()

#parser.add_argument("--name", required=True, type=str, help="model_name")
#parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)

generator = pipeline('text-generation',
                     model=model,
                     tokenizer=tokenizer,
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float16,
                                           replace_with_kernel_inject=True)


inputs = [
         "DeepSpeed is a machine learning framework",
         "He is working on",
         "He has a",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way"
]

outputs = generator(inputs,
        max_new_tokens=args.max_new_tokens
        )

if args.local_rank == 0:
    for i, o in zip(inputs, outputs):
        print(f"\nin={i}\nout={o}\n{'-'*60}")
    average_num_tokens = 0
    for o in outputs:
        for key, value in o[0].items():
            average_num_tokens += len(tokenizer.tokenize(value))
    average_num_tokens = average_num_tokens/len(inputs)
    print(average_num_tokens)


