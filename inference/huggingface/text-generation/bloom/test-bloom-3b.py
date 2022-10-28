import os
import torch
import deepspeed
import transformers

# import deepspeed pipeline helper
from utils import Pipeline

# import HF pipeline
#from transformers import pipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load-ckpt', action="store_true", help='enable meta tensor')
parser.add_argument('--model-name', type=str, default='bigscience/bloom-3b', help='model name')
parser.add_argument('--enable-meta-tensor', action="store_true", help='enable load ckpt')

args = parser.parse_args()

print(f"Running the {args.model_name} model with following settings:")
print(f"enable_meta_tensor: {args.enable_meta_tensor}")
print(f"enable_load_ckpt: {args.load_ckpt}")

model_name = args.model_name
dtype = torch.float16
num_tokens = 100

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

pipe = Pipeline(model_name=model_name, dtype=dtype, enable_meta_tensor=args.enable_meta_tensor)

ckpt_load = args.load_ckpt
if 'bloom' in model_name.lower():
    ckpt_load = True

if ckpt_load:
    kwargs = dict(base_dir=pipe.repo_root, 
                  checkpoint=pipe.checkpoints_json)
else:
    kwargs = dict()
    
pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=dtype,
    replace_with_kernel_inject=True,
    **kwargs
)   # base_dir=pipe.repo_root,
    # checkpoint=pipe.checkpoints_json


output = pipe('DeepSpeed is', num_tokens=num_tokens)
print(output)
