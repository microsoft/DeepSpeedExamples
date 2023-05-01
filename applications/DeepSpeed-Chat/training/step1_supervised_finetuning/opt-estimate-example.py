import torch
from deepspeed.utils import OnDevice
from transformers import AutoConfig, AutoModelForCausalLM


# model_name = 'facebook/opt-1.3b'
model_name = "stabilityai/stablelm-base-alpha-3b"

config = AutoConfig.from_pretrained(model_name)

# adjust to 175B size
# https://github.com/facebookresearch/metaseq/blob/eca010efc48be4ec418eb70349c7e68d3886cec5/metaseq/launcher/opt_job_constants.py#L44
#config.num_hidden_layers = 96
#config.hidden_size = 12288
#config.num_attention_heads = 96
#config.ffn_dim = 12288 * 4
#config.word_embed_proj_dim = 12288

print(config)

# replace OneDevice w. zero.Init if trying to add this into real code
#with deepspeed.zero.Init():
with OnDevice(dtype=torch.float16, device='meta'):
    model = AutoModelForCausalLM.from_config(config)

print(model.config)

num_params = sum([p.numel() for p in model.parameters()])
print(f"number of parameters: {num_params}")


layers = config.num_hidden_layers
hd = config.hidden_size
seq = 512 # args.max_seq_len
batch = 2 # args.per_device_train_batch_size
vocab = config.vocab_size
heads = config.num_attention_heads
# TODO(ammar): need to add intermediate size to config, llama will require this

scale = 1000000000
scale = 1

# =9*I18*I19*I20*I17*2/1000000000
gemms = 9 * hd * seq * batch * layers * 2 / scale
print(f"{gemms=} GB")

# =2*I20*I19*I19*I22*I17*2/1000000000
attn = 2 * batch * seq * seq * heads * layers * 2 / scale
print(f"{attn=} GB")

# =2*I19*I20*I18*I17*2/1000000000
ln = 2 * seq * batch * hd * layers * 2 / scale
print(f"{ln=} GB")

# =4*I18*I20*I19*I17*2/1000000000
gelu = 4 * hd * batch * seq * layers * 2 / scale
print(f"{gelu=} GB")

# =2 *I20*I19*I21*2/1000000000
loss = 2 * batch * seq * vocab * 2 / scale
print(f"{loss=} GB")

total = gemms + attn + ln + gelu + loss

print("Total activation memory required: {:,} Bytes".format(total))

# s * b * h * L * (10 + (24 / t) + 5 * ((a * s) / (h * t)) )
# tp = 1
# act_mem = seq * batch * hd * layers #* (10 + (24 / tp) + 5 * ((attn * seq) / (hd * tp)))
# tmp = 10 + (24 / tp)
# # tmp2 = (attn * seq) / (hd * tp)
# act_mem *= (tmp) # + 5 * tmp2)

act_mem = seq * batch * hd * layers * (34 + 5 * ((heads * seq) / hd))

print("Quentin's activation memory required: {:,} Bytes".format(act_mem))

# sbh2L

act_mem_ckpt = seq * batch * hd * 2 * layers

print("Quentin's activation memory (ckpt) required: {:,} Bytes".format(act_mem_ckpt))

#num_params * 2 * trainable_params * 12

## Samyam's formula
# sbh(10+24) + sbhL*2
act_mem = seq * batch * hd * (10 + 24) + seq * batch * hd * layers * 2
print("Samyam's activation memory required: {:,} Bytes".format(act_mem))
