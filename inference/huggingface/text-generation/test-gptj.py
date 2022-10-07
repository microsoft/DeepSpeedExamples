import os
import torch
import deepspeed
import transformers

from deepspeed import module_inject
from transformers import pipeline

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

generator = pipeline('text-generation',
                     model="EleutherAI/gpt-j-6B",
                     tokenizer="EleutherAI/gpt-j-6B",
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                 mp_size=world_size,
                                 dtype=torch.half,
                                 replace_with_kernel_inject=True)
                                 
string = generator("DeepSpeed is", min_length=50, max_length=50, do_sample=True, use_chache=True)
print(string)
