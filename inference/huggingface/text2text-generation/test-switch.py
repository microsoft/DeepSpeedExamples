from transformers import pipeline, T5Tokenizer, SwitchTransformersModel, SwitchTransformersForConditionalGeneration
import transformers
import deepspeed
import torch
import os
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersBlock, SwitchTransformersStack

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

tokenizer = T5Tokenizer.from_pretrained("google/switch-base-8")
model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8")

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=local_rank)

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float16,
    injection_policy={SwitchTransformersStack: ('SelfAttention.o', 'EncDecAttention.o', 'mlp.wo', 
        #'router.classifier',
        'expert_0.wo','expert_1.wo','expert_2.wo','expert_3.wo','expert_4.wo','expert_5.wo','expert_6.wo', 
        'expert_7.wo'
        )}
)

#print(pipe.model)

pipe.device = torch.device(f'cuda:{local_rank}')
output = pipe("summarize: Studies have been shown that owning a dog is good for you")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
