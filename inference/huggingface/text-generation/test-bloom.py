import os
import torch
import deepspeed
import transformers

from deepspeed import module_inject
#from transformers import pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers.models.bloom.modeling_bloom import BloomBlock

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-3b")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b")

generator = pipeline('text-generation',
                     #model='bigscience/bloom-3b',
                     model=model,
                     tokenizer=tokenizer,
#                     model='EleutherAI/gpt-neo-2.7B',
                     device=local_rank)
generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
                                           replace_with_kernel_inject=True)

# generator.model = deepspeed.init_inference(generator.model,
#                                            mp_size=world_size,
#                                            dtype=torch.float,
#                                            injection_policy={BloomBlock: ('self_attention.query_key_value')}
# )

# generator.model = deepspeed.init_inference(generator.model,
#                                            mp_size=world_size,
#                                            dtype=torch.float)

string = generator("DeepSpeed is", do_sample=True, min_length=50, max_length=50)
print(string)
