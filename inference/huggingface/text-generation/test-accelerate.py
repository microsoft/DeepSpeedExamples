import os
import math
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

input_sentences = [
         "He is working on",
         "DeepSpeed is a machine learning framework",
         "He got all",
         "Everyone is happy and I can",
         "The new movie that got Oscar this year",
         "In the far far distance from our galaxy,",
         "Peace is the only way"
]

batch_size = 3

if batch_size > len(input_sentences):
    # dynamically extend to support larger bs by repetition
    input_sentences *= math.ceil(batch_size / len(input_sentences))

inputs = input_sentences[:batch_size]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(inputs, return_tensors="pt", padding=True)

output = model.generate(inputs["input_ids"].to(0))
outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
print(outputs)
