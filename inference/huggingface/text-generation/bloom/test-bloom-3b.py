import os
import torch
import deepspeed
import transformers

# BloomPipeline class to mimic HF pipeline
from utils import BloomPipeline

model_name = 'bigscience/bloom-3b'
dtype = torch.float16
num_tokens = 100

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

output = pipe('DeepSpeed is', num_tokens=num_tokens)
print(output)
