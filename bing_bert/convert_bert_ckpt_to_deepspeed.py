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

def load_tf_weights_in_bert(model, config, ckpt_path, voc_size_diff):
    """ Load tf checkpoints in DeepSpeed model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in DeepSpeed, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(ckpt_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    qkv = {}
    for name_str, array in zip(names, arrays):
        name = name_str.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        key = None
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            # Special in deepspeed.
            elif name_str.find("bert/pooler/dense") >= 0 and scope_names[0] == "dense":
                pointer = getattr(pointer, "dense_act")
            elif name_str.find("bert/embeddings/LayerNorm/gamma") >= 0 and scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif name_str.find("bert/embeddings/LayerNorm/beta") >= 0 and scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif name_str.find("cls/predictions/transform/LayerNorm/gamma") >= 0 and scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif name_str.find("cls/predictions/transform/LayerNorm/beta") >= 0 and scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif name_str.find("cls/predictions/transform/dense") >= 0 and scope_names[0] == "dense":
                pointer = getattr(pointer, "dense_act")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])

                pointer = pointer[num]

                # For transofrmer kernel layers.
                if scope_names[0] == 'layer':
                    if name_str.find("attention/self/query/kernel") > 0:
                        key = "qw"
                    elif name_str.find("attention/self/query/bias") > 0:
                        key = "qb"
                    elif name_str.find("attention/self/key/kernel") > 0:
                        key = "kw"
                    elif name_str.find("attention/self/key/bias") > 0:
                        key = "kb"
                    elif name_str.find("attention/self/value/kernel") > 0:
                        key = "vw"
                    elif name_str.find("attention/self/value/bias") > 0:
                        key = "vb"
                    elif name_str.find("attention/output/dense/kernel") > 0:
                        pointer = getattr(pointer, "attn_ow")
                    elif name_str.find("attention/output/dense/bias") > 0:
                        pointer = getattr(pointer, "attn_ob")
                    elif name_str.find("attention/output/LayerNorm/gamma") > 0:
                        pointer = getattr(pointer, "attn_nw")
                    elif name_str.find("attention/output/LayerNorm/beta") > 0:
                        pointer = getattr(pointer, "attn_nb")
                    elif name_str.find("intermediate/dense/kernel") > 0:
                        pointer = getattr(pointer, "inter_w")
                    elif name_str.find("intermediate/dense/bias") > 0:
                        pointer = getattr(pointer, "inter_b")
                    elif name_str.find("output/dense/kernel") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "output_w")
                    elif name_str.find("output/dense/bias") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "output_b")
                    elif name_str.find("output/LayerNorm/gamma") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "norm_w")
                    elif name_str.find("output/LayerNorm/beta") > 0 and name_str.find("attention") < 0:
                        pointer = getattr(pointer, "norm_b")
                    else:
                        raise ValueError(f"unexpect scope name {name_str} in transformer layer.")
                    break

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif "kernel" in name:
            array = np.transpose(array)

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
            # For Q/K/V weight/bias in TF, do nothing if not all ready to merge.
            continue

        # DeepSpeed BERT model has voc_size 8 aligned.
        if voc_size_diff > 0 and name_str.find("embeddings/word_embeddings") >= 0:
            z = np.zeros((voc_size_diff, array.shape[1]), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)
        if voc_size_diff > 0 and name_str.find("cls/predictions/output_bias") >= 0:
            z = np.zeros((voc_size_diff), dtype=array.dtype)
            array = np.concatenate((array, z), axis=0)

        set_data(pointer, array)
        logger.info("Initialize DeepSpeed weight {}".format(name))

    return model

def load_hf_weights_in_bert(model, config, ckpt_path, voc_size_diff):
    """ Load huggingface checkpoints and convert to a deepspeed model.
    """
    hf_path = os.path.abspath(ckpt_path)
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

def convert_ckpt_to_deepspeed(ckpt_type, ckpt_path, bert_config_file, deepspeed_dump_dir, args):
    # Initialise DeepSpeed model
    config = BertConfig.from_json_file(bert_config_file)

    # DeepSpeed BERT model has voc_size 8 aligned.
    orig_voc_size = config.vocab_size
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    voc_size_diff = config.vocab_size - orig_voc_size

    print("Building DeepSpeed model from configuration: {}".format(str(config)))
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

    # Load weights from checkpoint
    if ckpt_type == "HF":
        load_hf_weights_in_bert(model.module, config, ckpt_path, voc_size_diff )
    elif ckpt_type == "TF":
        load_tf_weights_in_bert(model.module, config, ckpt_path, voc_size_diff )
    else:
        raise ValueError(f"Invalid ckpt_type.")

    print("Save DeepSpeed model to {}".format(deepspeed_dump_dir))
    success = model.save_checkpoint(deepspeed_dump_dir, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--ckpt_type", type=str, required=True, help="Checkpoint's type, TF - Tensorflow, HF - Huggingface.")

    parser.add_argument(
        "--ckpt_path", default=None, type=str, required=True, help="Path to the checkpoint file."
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
        "--deepspeed_dump_dir", default=None, type=str, required=True, help="Path to the output DeepSpeed model."
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

    convert_ckpt_to_deepspeed(args.ckpt_type, args.ckpt_path, args.bert_config_file, args.deepspeed_dump_dir , args)
