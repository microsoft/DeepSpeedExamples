import os
import torch
import deepspeed
import transformers
from pathlib import Path

from deepspeed import module_inject
from transformers import pipeline
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock as gpt2_transformer
from ort_utils import create_model_for_provider, inference_onnx_binding, optimize_onnx
from pytorch_utils import convert_to_onnx, get_model_size
from trt_utils import build_engine, load_engine, save_engine
from transformers import AutoConfig, AutoTokenizer, BatchEncoding, PretrainedConfig, PreTrainedTokenizer, TensorType

from itertools import chain
from torch.onnx import export
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.onnx.features import FeaturesManager
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoModel,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)
from deepspeed import module_inject
from transformers import pipeline
from transformers.models.gptj.modeling_gptj import GPTJBlock

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

# model_name_or_path = "EleutherAI/gpt-neo-2.7B"
model_name_or_path = "EleutherAI/gpt-j-6B"


generator = pipeline('text-generation',
                     model=model_name_or_path,
                     device=local_rank)

import time

string = generator("Wikipedia is", do_sample=True, min_length=50, max_length=50)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    generator.model = deepspeed.init_inference(generator.model,
                                            mp_size=world_size,
                                            dtype=torch.float,
                                            replace_method='auto',
                                            replace_with_kernel_inject=True)
    # generator.model = deepspeed.init_inference(generator.model,
    #                                 mp_size=world_size,
    #                                 dtype=torch.float,
    #                                 injection_policy={GPTJBlock: ('attn.out_proj','mlp.fc_out')},
    #                                 replace_with_kernel_inject=False)

    string = generator("Wikipedia is", do_sample=True, min_length=50, max_length=50)
print(time.time()-t0)
torch.cuda.synchronize()

print(string)