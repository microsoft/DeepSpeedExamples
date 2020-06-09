# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert BERT checkpoint."""


import os
import argparse
import logging
import torch
import deepspeed
import re
import numpy as np
from nvidia.modelingpreln import BertForPreTrainingPreLN, BertConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_data(param, array):
    try:
        assert param.shape == array.shape
    except AssertionError as e:
        e.args += (param.shape, array.shape)
        raise
    param.data = torch.from_numpy(array)


def load_hf_weights_in_bert(model, config, hf_checkpoint_path, voc_size_diff):
    """ Load huggingface checkpoints and convert to a deepspeed model.
    """
    hf_path = os.path.abspath(hf_checkpoint_path)
    logger.info("Converting Huggingface checkpoint from {}".format(hf_path))
    # Load weights from Huggingface model
    ckpt = torch.load(hf_path, map_location=torch.device("cpu"))

    qkv = {}
    for name_str in ckpt.keys():
        array = ckpt[name_str].numpy()
        logger.info("Loading Huggingface weight {} with shape {}".format(name_str, array.shape))
        name = name_str.split(".")
        pointer = model
        key = None
        is_layer = False
        for m_name in name:
            # Special in deepspeed.
            if name_str.find("bert.pooler.dense") >= 0 and m_name == "dense":
                pointer = getattr(pointer, "dense_act")
            elif name_str.find("cls.predictions.transform.dense") >= 0 and m_name == "dense":
                pointer = getattr(pointer, "dense_act")
            elif is_layer:
                pass
                #import pdb; pdb.set_trace()
            else:
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    logger.info("Skipping {}".format(".".join(name)))
                    continue

            if m_name == "layer":
                is_layer = True
                continue

            if m_name.isnumeric() and is_layer:
                num = int(m_name)
                pointer = pointer[num]
                is_layer = False

                # For transofrmer kernel layers.
                if name_str.find("attention.self.query.weight") > 0:
                    key = "qw"
                elif name_str.find("attention.self.query.bias") > 0:
                    key = "qb"
                elif name_str.find("attention.self.key.weight") > 0:
                    key = "kw"
                elif name_str.find("attention.self.key.bias") > 0:
                    key = "kb"
                elif name_str.find("attention.self.value.weight") > 0:
                    key = "vw"
                elif name_str.find("attention.self.value.bias") > 0:
                    key = "vb"
                elif name_str.find("attention.output.dense.weight") > 0:
                    pointer = getattr(pointer, "attn_ow")
                elif name_str.find("attention.output.dense.bias") > 0:
                    pointer = getattr(pointer, "attn_ob")
                elif name_str.find("attention.output.LayerNorm.weight") > 0:
                    pointer = getattr(pointer, "attn_nw")
                elif name_str.find("attention.output.LayerNorm.bias") > 0:
                    pointer = getattr(pointer, "attn_nb")
                elif name_str.find("intermediate.dense.weight") > 0:
                    pointer = getattr(pointer, "inter_w")
                elif name_str.find("intermediate.dense.bias") > 0:
                    pointer = getattr(pointer, "inter_b")
                elif name_str.find("output.dense.weight") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "output_w")
                elif name_str.find("output.dense.bias") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "output_b")
                elif name_str.find("output.LayerNorm.weight") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "norm_w")
                elif name_str.find("output.LayerNorm.bias") > 0 and name_str.find("attention") < 0:
                    pointer = getattr(pointer, "norm_b")
                else:
                    raise ValueError(f"unexpect scope name {name_str} in transformer layer.")
                break

        if key is not None:
            qkv[key] = array

        if all(k in qkv for k in ("qw", "kw", "vw")):
            array = np.concatenate((qkv["qw"], qkv["kw"], qkv["vw"]), axis=0)
            pointer = getattr(pointer, "attn_qkvw")
            qkv.pop("qw")
            qkv.pop("kw")
            qkv.pop("vw")
        elif all(k in qkv for k in ("qb", "kb", "vb")):
            array = np.concatenate((qkv["qb"], qkv["kb"], qkv["vb"]), axis=0)
            pointer = getattr(pointer, "attn_qkvb")
            qkv.pop("qb")
            qkv.pop("kb")
            qkv.pop("vb")
        elif key is not None:
            # For Q/K/V weight/bias in HF, do nothing if not all ready to merge.
            continue

        # DeepSpeed BERT model has voc_size 8 aligned.
        if voc_size_diff > 0 and name_str.find("embeddings.word_embeddings") >= 0:
            z = np.zeros((voc_size_diff, array.shape[1]), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)
        if voc_size_diff > 0 and name_str.find("cls.predictions.bias") >= 0:
            z = np.zeros((voc_size_diff), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)
        if voc_size_diff > 0 and name_str.find("cls.predictions.decoder.weight") >= 0:
            z = np.zeros((voc_size_diff, array.shape[1]), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)

        set_data(pointer, array)
        logger.info("Initialize DeepSpeed weight {}".format(name))

    return model

def convert_hf_ckpt_to_deepspeed(hf_checkpoint_path, bert_config_file, deepspeed_dump_dir, args):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)

    # DeepSpeed BERT model has voc_size 8 aligned.
    orig_voc_size = config.vocab_size
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    voc_size_diff = config.vocab_size - orig_voc_size

    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTrainingPreLN(config, args)
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    model, _, _, _ = deepspeed.initialize(args=args,
                                          model=model,
                                          model_parameters=optimizer_grouped_parameters )
    print(model)

    # Load weights from checkpoint
    load_hf_weights_in_bert(model.module, config, hf_checkpoint_path, voc_size_diff )

    # Save pytorch-model
    print("Save PyTorch model to {}".format(deepspeed_dump_dir))
    success = model.save_checkpoint(deepspeed_dump_dir, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--hf_checkpoint_path", default=None, type=str, required=True, help="Path to the Huggingface checkpoint path."
    )

    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )

    parser.add_argument(
        "--deepspeed_dump_dir", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )

    parser.add_argument("--deepspeed_transformer_kernel",
                        default=True,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel to accelerate.')

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--attention_dropout_checkpoint',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel memory optimization to checkpoint dropout output.')

    parser.add_argument('--normalize_invertible',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel memory optimization to perform invertible normalize backpropagation.')

    parser.add_argument('--gelu_checkpoint',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel memory optimization to checkpoint GELU activation.')

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--stochastic_mode',
                        default=False,
                        action='store_true',
                        help='Use stochastic mode for high-performance transformer kernel.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    convert_hf_ckpt_to_deepspeed(args.hf_checkpoint_path, args.bert_config_file, args.deepspeed_dump_dir , args)
