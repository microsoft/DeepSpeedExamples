import os
import torch
import deepspeed
import transformers

# BloomPipeline class to mimic HF pipeline
from utils import BloomPipeline

batch_size = 8
num_tokens = 100

model_name = 'bigscience/bloom-3b'
dtype = torch.float16

input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]

if batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(batch_size / len(input_sentences))

inputs = input_sentences[: batch_size]

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

pipe = BloomPipeline(model_name=model_name,
                    dtype=dtype)

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=dtype,
    replace_with_kernel_inject=True,
    base_dir=pipe.repo_root,
    checkpoint=pipe.checkpoints_json
)

outputs = pipe(inputs, num_tokens=num_tokens)

for i, o in zip(inputs, outputs):
    print(f"{'-'*60}\nin={i}\nout={o}\n")
