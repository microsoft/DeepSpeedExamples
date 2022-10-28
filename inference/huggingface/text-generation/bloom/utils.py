'''
Helper classes and functions for the BLOOM examples
'''

from cgitb import enable
import io
from pathlib import Path
import json
import deepspeed
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class Pipeline():
    '''Pipeline helper class as an alternative to HF pipelines for models that are loaded with meta tensors and checkpoints'''
    def __init__(self,
                 model_name='gpt2',
                 dtype=torch.float16,
                 enable_meta_tensor=True):
        self.model_name = model_name
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.repo_root, self.checkpoints_json = self.generate_json()
        self.enable_meta_tensor = enable_meta_tensor
        
        # Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
        if self.enable_meta_tensor:
            with deepspeed.OnDevice(dtype=self.dtype, device="meta"):
                self.model = AutoModelForCausalLM.from_config(self.config, torch_dtype=self.dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.model.eval()

    def __call__(self,
                inputs=["test"],
                num_tokens=100):
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs
        outputs = self.generate_outputs(input_list, num_tokens=num_tokens)
        return outputs

    def generate_json(self):
        repo_root = snapshot_download(self.model_name, allow_patterns=["*"], local_files_only=False, revision=None)

        checkpoints_json = "checkpoints.json"

        with io.open(checkpoints_json, "w", encoding="utf-8") as f:
            file_list = [str(entry) for entry in Path(repo_root).rglob("*.[bp][it][n]") if entry.is_file()]
            data = {"type": self.config.model_type, "checkpoints": file_list, "version": 1.0}
            json.dump(data, f)

        return repo_root, checkpoints_json

    def generate_outputs(self,
                         inputs=["test"],
                         num_tokens=100):
        generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_tokens = self.tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

        outputs = self.model.generate(**input_tokens, **generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs
