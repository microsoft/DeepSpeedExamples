import os
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

checkpoint = "EleutherAI/gpt-j-6B"
weights_path = snapshot_download(checkpoint)
files = os.listdir(weights_path)
weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model = load_checkpoint_and_dispatch(
    model, weights_path, device_map="auto", no_split_module_classes=["GPTJBlock"]
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer("Hello, my name is", return_tensors="pt")
inputs = inputs.to(0)
output = model.generate(inputs["input_ids"])
print(tokenizer.decode(output[0].tolist()))
