from transformers import pipeline
import transformers
import deepspeed
import torch
import os
from transformers.models.mvp.modeling_mvp import MvpDecoderLayer, MvpEncoderLayer

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

pipe = pipeline("text2text-generation", model="RUCAIBox/mvp", device=local_rank)

# The injection_policy shows two things:
#   1. which layer module we need to add Tensor-Parallelism
#   2. the name of several linear layers: a) attention_output (both encoder and decoder),
#       and b) transformer output

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={MvpDecoderLayer: ('self_attn.out_proj', 'encoder_attn.out_proj', '.fc2'), MvpEncoderLayer: ('self_attn.out_proj', '.fc2')}
)

pipe.device = torch.device(f'cuda:{local_rank}')
output = pipe("Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
