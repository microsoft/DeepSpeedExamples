import deepspeed
import torch
import os
from transformers import pipeline, T5Tokenizer, T5Model

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

# Init translator
translator = pipeline("translation_en_to_fr", model="t5-base", tokenizer="t5-base", device=local_rank)

# DeepSpeed init_inference API
translator.model = deepspeed.init_inference(translator.model,
                                           mp_size=world_size,
                                           dtype=torch.float16,
                                           replace_with_kernel_inject=True)

# Translate text
text = "The quick brown fox jumps over the lazy dog."
translation = translator(text)

# Print translation
print(translation)
