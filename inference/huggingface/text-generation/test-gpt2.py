import os
import torch
import deepspeed
import transformers

from deepspeed import module_inject
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

generator = pipeline('text-generation',
                     model=model,
                     tokenizer=tokenizer,
                     device=local_rank)

#config = deepspeed.default_inference_config()
from rich.pretty import pprint
#pprint(config["mp_size"])
#pprint(config)
#config["mp_size"] = world_size
#pprint(config["mp_size"])

#myconf = { "replace_with_kernel_inject" : True, "dtype": torch.half, "mp_size" : world_size }

#generator.model = deepspeed.init_inference(generator.model, config=config)
#string = generator("DeepSpeed is", min_length=50, max_length=50, do_sample=False, use_cache=True)
#print(string)


#generator.model = deepspeed.init_inference(generator.model, config=myconf, replace_with_kernel_inject=True)
#string = generator("DeepSpeed is", min_length=50, max_length=50, do_sample=False, use_cache=True)
#print(string)
#
#
## New API usage - no config and no kwargs
#generator.model = deepspeed.init_inference(generator.model)
#string = generator("DeepSpeed is", min_length=50, max_length=50, do_sample=False, use_cache=True)
#print(string)

# Existing usage
generator.model = deepspeed.init_inference(generator.model,
                                 mp_size=world_size,
                                 dtype=torch.half,
                                 replace_with_kernel_inject=True)
string = generator("DeepSpeed is", min_length=50, max_length=50, do_sample=False)
print(string)


