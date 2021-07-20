import os
import torch
import deepspeed
import transformers

from deepspeed import module_inject
from transformers import pipeline
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock as gpt2_transformer

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

generator = pipeline('text-generation',
                     model='EleutherAI/gpt-neo-2.7B')
print(
    f'({local_rank}) before deepspeed: {torch.cuda.memory_allocated()}, {torch.cuda.memory_cached()}'
)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.half,
                                           replace_method='auto')
generator.device = torch.device(f'cuda:{local_rank}')

print(
    f'({local_rank}) after deepspeed: {torch.cuda.memory_allocated()}, {torch.cuda.memory_cached()}'
)
string = generator("DeepSpeed is", do_sample=True, min_length=50)
print(f'[{local_rank}]: {string}')
