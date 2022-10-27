import os
import io
from pathlib import Path
import json
import torch
import deepspeed
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

batch_size = 8
num_tokens = 100

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

repo_root = snapshot_download('bigscience/bloom-3b', allow_patterns=["*"], local_files_only=False, revision=None)

checkpoints_json = "checkpoints.json"

with io.open(checkpoints_json, "w", encoding="utf-8") as f:
    file_list = [str(entry) for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
    data = {"type": "BLOOM", "checkpoints": file_list, "version": 1.0}
    json.dump(data, f)

tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-3b')
config = AutoConfig.from_pretrained('bigscience/bloom-3b')

# Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

model.eval()

model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    base_dir=repo_root,
    checkpoint=checkpoints_json
)

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

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
for t in input_tokens:
    if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

outputs = model.generate(**input_tokens, **generate_kwargs)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, o in zip(inputs, outputs):
    print(f"{'-'*60}\nin={i}\nout={o}\n")
