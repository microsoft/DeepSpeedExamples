from argparse import ArgumentParser
import transformers
import deepspeed
import torch
import os
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from deepspeed.runtime.utils import see_memory_usage

class DSPipeline():
    def __init__(self,
                 model_name='t5-11b',
                 dtype=torch.float16,
                 device=-1,
                 checkpoint_path=None
                 ):
        self.model_name = model_name
        self.dtype = dtype

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", model_max_length=50)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()

    def __call__(self,
                inputs=["test"],
                ):
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs
        outputs = self.generate_outputs(input_list)
        return outputs


    def generate_outputs(self,
                         inputs=["test"],

                        ):
        generate_kwargs = dict(max_new_tokens=50)
        input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)
        self.model.cuda().to(self.device)
        outputs = self.model.generate(**input_tokens,**generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs


parser = ArgumentParser()
parser.add_argument("--model", required=True, type=str, help="model_id")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
args = parser.parse_args()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '4'))

pipe = DSPipeline(model_name=args.model,
                  dtype=torch.float16,
                  device=args.local_rank,
                  )

if local_rank == 0:
    see_memory_usage("before init", True)

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float16,
)

if local_rank == 0:
    see_memory_usage("after init", True)

pipe.device = torch.device(f'cuda:{local_rank}')
output = pipe("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
