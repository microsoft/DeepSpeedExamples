#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

# Updated from HuggingFace Transformers commit d9c62047a8d75e18d2849d345ab3394875a712ef


import argparse
import logging
import time
import numpy as np
import torch
from pathlib import Path
from typing import Callable, Dict
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as gpt2_transformer
from transformers import (
    AutoModelForCausalLM,
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoModel,
    GPTNeoForCausalLM,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import torch
from torch.nn import Module
from transformers import AutoConfig, AutoTokenizer, BatchEncoding, PretrainedConfig, PreTrainedTokenizer, TensorType


# refer to https://github.com/ELS-RD/transformer-deploy/blob/main/src/transformer_deploy/convert.py
from transformer_deploy.backends.ort_utils import (
    cpu_quantization,
    create_model_for_provider,
    inference_onnx_binding,
    optimize_onnx,
)
from transformer_deploy.backends.pytorch_utils import (
    convert_to_onnx,
    get_model_size,
    infer_classification_pytorch,
    infer_feature_extraction_pytorch,
)

from itertools import chain
from torch.onnx import export
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.onnx.features import FeaturesManager
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gptneo": (GPTNeoForCausalLM, GPT2Tokenizer),
    "gptj":(AutoModelForCausalLM, AutoTokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}


# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def print_latency(latency_set, title=""):
    # 10 warmup queries
    latency_set = latency_set[10:]
    count = len(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print("====== latency stats {0} ======", title)
        print("\tAvg Latency: {0:8.2f} ms".format(avg * 1000))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 * 1000))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 * 1000))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 * 1000))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 * 1000))
        print("\t999 Latency: {0:8.2f} ms".format(p999 * 1000))
        print(f"{avg * 1000}, {p50 * 1000}, {p90 * 1000}, {p95 * 1000}, {p99 * 1000}, {p999 * 1000}")
class GPTModelWrapper(Module, GenerationMixin):
    def __init__(
        self, config: PretrainedConfig, device: torch.device, inference: Callable[[torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        self.config: PretrainedConfig = config
        self.device: torch.device = device
        self.inference: Callable[[torch.Tensor], torch.Tensor] = inference
        self.main_input_name = "input_ids"  # https://github.com/huggingface/transformers/pull/14803

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            self.main_input_name: input_ids,
        }

    def forward(self, input_ids, **_):
        logits = self.inference(input_ids)
        return CausalLMOutputWithCrossAttentions(logits=logits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--sample_input",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Whether to use int8-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument('--ds-inference', action="store_true", help="Use deepspeed")
    parser.add_argument('--ort', action="store_true", help="Use ORT")
    # parser.add_argument('--ort-cache', action="store_true", help="Use ORT")
    parser.add_argument('--trt', action="store_true", help="Use TRT")
    parser.add_argument('--base', action="store_true", help="Use pytorch baseline")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(
        "device: %s, n_gpu: %s, 16-bits: %s",
        args.device,
        args.n_gpu,
        args.fp16,
    )

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    auth_token = None
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=auth_token)
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path, use_auth_token=auth_token
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, use_auth_token=auth_token)
    model.eval()

    task= "text-generation"
    model_name_or_path = "tmp-"+ (args.model_name_or_path).replace("/", "-")
    if not Path(model_name_or_path).exists():
        os.makedirs(model_name_or_path)
    prefix = os.path.join(model_name_or_path, (args.model_name_or_path).replace("/", "-") + f"_fp16={args.fp16}_")
    ort_model_path = prefix  + ".onnx"
    ort_opt_model_path = prefix + "-opt.onnx"
    trt_model_path = prefix +".engine"

    # initialize deepspeed engine
    if args.ds_inference:
        model.cuda(torch.cuda.current_device())
        if args.fp16:
            model.half()

        import deepspeed.module_inject as module_inject
        import deepspeed

        dtype = (torch.half if args.fp16 else torch.float)
        if args.int8:
            dtype = torch.int8
        model = deepspeed.init_inference(model,
                                         mp_size=1,
                                         dtype=dtype,
                                         replace_method='auto',
                                         replace_with_kernel_inject=True)
        model = model.module

    elif args.ort:

        if not Path(ort_model_path).exists():
            input_ids: BatchEncoding = tokenizer(
            "Here is some text to encode Hello World", add_special_tokens=True, return_attention_mask=False, return_tensors="pt"
            )
            # some inference engines don't support int64 tensor as inputs, we convert all input tensors to int32 type
            for k, v in input_ids.items():
                if not isinstance(v, torch.Tensor):
                    continue
                if v.dtype in [torch.long, torch.int64]:
                    input_ids[k] = v.type(torch.int32)

            # create onnx model and compare results
            convert_to_onnx(
                model_pytorch=model,
                output_path=ort_model_path,
                inputs_pytorch=dict(input_ids),
                quantization=False,
                var_output_seq=(task in ["text-generation", "token-classification", "question-answering"]),
                output_names=["output"] if task != "question-answering" else ["start_logits", "end_logits"],
            )

            # model may switch to train mode for some unknown reasons, we force the eval mode.
        _ = model.eval()

        if not Path(ort_opt_model_path).exists():
            num_attention_heads, hidden_size = get_model_size(path=args.model_name_or_path)
            optimize_onnx(
                onnx_path=ort_model_path,
                ort_opt_model_path=ort_opt_model_path,
                fp16=True if args.fp16 else False,
                use_cuda=True,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                architecture="gpt2",
            )

        model_onnx = create_model_for_provider(path=ort_opt_model_path, provider_to_use="CUDAExecutionProvider")

        def inference_onnx_optimized(input_ids: torch.Tensor) -> torch.Tensor:
            input_ids = input_ids.type(dtype=torch.int32)

            data = {"input_ids": input_ids}
            return inference_onnx_binding(model_onnx=model_onnx, inputs=data, device="cuda")["output"]

        model = GPTModelWrapper(config=model.config, device=torch.device("cuda"), inference=inference_onnx_optimized)
    elif args.trt:
        if not Path(ort_model_path).exists():
            input_ids: BatchEncoding = tokenizer(
            "Here is some text to encode Hello World", add_special_tokens=True, return_attention_mask=False, return_tensors="pt"
            )
            # some inference engines don't support int64 tensor as inputs, we convert all input tensors to int32 type
            for k, v in input_ids.items():  # type: str, torch.Tensor
                input_ids[k] = v.type(dtype=torch.int32)

            # create onnx model and compare results
            convert_to_onnx(
                model_pytorch=model,
                output_path=ort_model_path,
                inputs_pytorch=dict(input_ids),
                quantization=False,
                var_output_seq=(task in ["text-generation", "token-classification", "question-answering"]),
                output_names=["output"] if task != "question-answering" else ["start_logits", "end_logits"],
            )
            print("Converted to ONNX")

        try:
            import tensorrt as trt
            from tensorrt.tensorrt import ICudaEngine, Logger, Runtime
            from transformer_deploy.backends.trt_utils import build_engine, load_engine, save_engine
        except ImportError:
            raise ImportError(
                "It seems that TensorRT is not yet installed. "
                "It is required when you declare TensorRT backend."
                "Please find installation instruction on "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
            )
        trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
        runtime: Runtime = trt.Runtime(trt_logger)

        max_seq_len = 128
        if not Path(trt_model_path).exists():
            print("Building TensorRT engine")
            engine: ICudaEngine = build_engine(
                runtime=runtime,
                onnx_file_path=ort_model_path,
                logger=trt_logger,
                min_shape=(1, 1),
                optimal_shape=(1, max_seq_len),  # num beam, batch size
                max_shape=(1, max_seq_len),  # num beam, batch size
                workspace_size=10000 * 1024**2,
                fp16=True if args.fp16 else False,
                int8=False,
            )
            save_engine(engine, trt_model_path)

        tensorrt_model: Callable[[Dict[str, torch.Tensor]], torch.Tensor] = load_engine(
            engine_file_path=trt_model_path, runtime=runtime
        )

        def inference_tensorrt(input_ids: torch.Tensor) -> torch.Tensor:
            input_ids = input_ids.type(dtype=torch.int32)
            data = {"input_ids": input_ids}
            return tensorrt_model(data)["output"]

        model = GPTModelWrapper(config=model.config, device=torch.device("cuda"), inference=inference_tensorrt)
    else:
        model.cuda(torch.cuda.current_device())

        if args.fp16:
            model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)
    if args.sample_input:
        fname = open(args.sample_input, "r", encoding="utf8")
        prompt_text = fname.readlines()
    else:
        prompt_text = (args.prompt if args.prompt else input("Model prompt >>> "),)

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    eprompt = []
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
        for input_text in prompt_text:
            preprocessed_prompt_text.append(prepare_input(args, model, tokenizer, prompt_text))

            if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}
            for ppt in preprocessed_prompt_text:
                eprompt.append(tokenizer.encode(
                    ppt, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
                ))
    else:
        prefix = args.prefix if args.prefix else args.padding_text
        for ppt in prompt_text:
            eprompt.append(tokenizer.encode(prefix + ppt, add_special_tokens=False, return_tensors="pt").to(torch.int32))

    latencies = []
    for encoded_prompt, ppt in zip(eprompt, prompt_text):
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        input_ids = input_ids.type(dtype=torch.int32)

        torch.cuda.synchronize()
        t0 = time.time()

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
            use_cache=True,
        )
        torch.cuda.synchronize()
        latencies.append((time.time()-t0) / output_sequences.numel())

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                ppt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            print(total_sequence)
    print_latency(latencies)
    return generated_sequences


if __name__ == "__main__":
    main()
