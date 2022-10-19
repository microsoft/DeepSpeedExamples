import os, torch
from statistics import mode
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

model_name = "EleutherAI/gpt-j-6B"

deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)

train_batch_size = world_size
tokenizer = AutoTokenizer.from_pretrained(model_name)

deepspeed_config = {
    "fp16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 3,
    },
    "train_batch_size": train_batch_size,
}
dschf = HfDeepSpeedConfig(deepspeed_config)

model = AutoModelForCausalLM.from_pretrained(model_name)

model = model.eval()
ds_engine = deepspeed.initialize(model=model, config_params=deepspeed_config)[0]
ds_engine.module.eval()
model = ds_engine.module

inp_tokens = tokenizer("DeepSpeed is", return_tensors="pt",)
for token in inp_tokens:
    if torch.is_tensor(inp_tokens[token]):
        inp_tokens[token] = inp_tokens[token].to(f'cuda:{local_rank}')

model.cuda().to(f'cuda:{local_rank}')
string = tokenizer.batch_decode(model.generate(**inp_tokens,min_length=50,max_length=50,do_sample=False))[0]
deepspeed.runtime.utils.see_memory_usage("after-generate-init", force=True)
print(string)
