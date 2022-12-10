import os
import torch
import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from deepspeed.model_implementations import DeepSpeedTransformerInference

##########################################
#           Static variables
##########################################
model_name = "EleutherAI/gpt-j-6B"
dtype = getattr(torch, 'int8')
model_tmpdir = './tmp/'


world_size = int(os.getenv("WORLD_SIZE", "1"))
local_rank = int(os.getenv('LOCAL_RANK', '0'))

device = torch.device(f"cuda:{local_rank}")

print(f"device = {device}")

##########################################
#           Save shard
##########################################
if not os.path.isdir(os.path.join(model_tmpdir)):
    inf_config = {
        "replace_with_kernel_inject": True,
        "dtype": torch.float16,
        "replace_method": "auto",
        "enable_cuda_graph": False,
        "tensor_parallel": {
            "tp_size": world_size
        },
        "save_mp_checkpoint_path": os.path.join(model_tmpdir),
    }

    # Load model and save sharded checkpoint
    model_save_shard = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16)

    model_save_shard = deepspeed.init_inference(model_save_shard, config=inf_config)

    # Get an inference error if we don't rerun script
    exit(0)


##########################################
#           Load checkpoint
##########################################
inf_config = {
    "replace_with_kernel_inject": True,
    "dtype": dtype,
    "replace_method": "auto",
    "enable_cuda_graph": False,
    "tensor_parallel": {
        "tp_size": world_size
    },
    "checkpoint": os.path.join(model_tmpdir,
                                "ds_inference_config.json"),
}


# Load model on meta tensors
model_config = AutoConfig.from_pretrained(model_name)
# Note that we use half precision to load initially, even for int8
with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
    model = AutoModelForCausalLM.from_config(model_config,
                                                torch_dtype=torch.bfloat16)
model = model.eval()
model = deepspeed.init_inference(model, config=inf_config)


##########################################
#           Check dtype
##########################################
def check_dtype(model, expected_dtype):
    def find_dtype(module):
        for child in module.children():
            if isinstance(child, DeepSpeedTransformerInference):
                return child.attention.attn_qkvw.dtype
            else:
                found_dtype = find_dtype(child)
                if found_dtype:
                    return found_dtype

    found_dtype = find_dtype(model)
    assert found_dtype, "Did not find DeepSpeedTransformerInference in model"
    assert (
        found_dtype == expected_dtype
    ), f"Expected transformer dtype {expected_dtype}, but found {found_dtype}"

check_dtype(model, dtype)

##########################################
#           Build tokenizer
##########################################
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token


##########################################
#           Define inputs
##########################################
inputs = [
         "DeepSpeed is a machine learning framework",
]

num_tokens = 50


##########################################
#           Generate outputs
##########################################
generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
for t in input_tokens:
    if torch.is_tensor(input_tokens[t]):
        input_tokens[t] = input_tokens[t].to(device)

model.cuda().to(device)

outputs = model.generate(**input_tokens, **generate_kwargs)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

for i, o in zip(inputs, outputs):
    print(f"\nin={i}\nout={o}\n{'-'*60}")
