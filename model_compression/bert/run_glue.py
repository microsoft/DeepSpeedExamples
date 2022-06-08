# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
from pathlib import Path
import copy
import datasets
from datasets import load_dataset, load_metric
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import time
import transformers
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    default_data_collator,
    get_scheduler,
    set_seed,
)
import json
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from huggingface_models.modeling_bert import BertForSequenceClassification
import deepspeed
from deepspeed.compression.helper import module_replacement, fix_compression, compression_preparation
from deepspeed.compression.basic_layer import LinearLayer_Compress
from deepspeed.compression.compress import student_initialization, compress, redundant_clean

logger = logging.getLogger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt"
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst2": "classification",
    "stsb": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification"
}
default_params = {
    "cola": {
        "max_seq_length": 64,
        "batch_size": 16,
        "eval_step": 50
    },
    "mnli": {
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 1000
    },
    "mrpc": {
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 200
    },
    "sst2": {
        "max_seq_length": 64,
        "batch_size": 32,
        "eval_step": 200
    },
    "stsb": {
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 50
    },
    "qqp": {
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 1000
    },
    "qnli": {
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 1000
    },
    "rte": {
        "max_seq_length": 128,
        "batch_size": 32,
        "eval_step": 50
    }
}
acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
corr_tasks = ["sts-b"]
mcc_tasks = ["cola"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.")
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=
        ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
         " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
         ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help=
        "If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_teacher",
        default=None,
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help=
        "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=
        "Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--distill_method",
                        type=str,
                        default="three_stage",
                        help="Where to store the final model.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the final model.")
    parser.add_argument("--pretrained_dir_student",
                        type=str,
                        default=None,
                        help="Where to load the pretrained model.")
    parser.add_argument("--pretrained_dir_teacher",
                        type=str,
                        default=None,
                        help="Where to load the pretrained model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # parser.add_argument("--layer_reductio_enabled", action="store_true", help="reduce layer")
    # parser.add_argument("--prune_enabled", action="store_true", help="prune model")
    # parser.add_argument("--quantization_enabled", action="store_true", help="quantize model")
    parser.add_argument("--weight_bit",
                        type=int,
                        default=None,
                        help="weight bit.")
    parser.add_argument("--save_best_checkpoint",
                        action="store_true",
                        help="save best checkpoint model")
    parser.add_argument("--save_last_model",
                        action="store_true",
                        help="save the last model")
    parser.add_argument("--clean_last_model",
                        action="store_true",
                        help="clean the last model")
    parser.add_argument("--deepspeed",
                        action="store_true",
                        help="use deepspeed or not")
    parser.add_argument("--deepspeed_config",
                        type=str,
                        default=None,
                        help="deepspeed config")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError(
            "Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv", "json"
            ], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv", "json"
            ], "`validation_file` should be a csv or a json file."
    return args


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (-targets_prob * student_likelihood).mean()


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    args = parse_args()
    # ds_config = DeepSpeedConfig(args.deepspeed_config)
    if args.deepspeed:
        with open(args.deepspeed_config) as f:
            ds_config = json.load(f)
        assert args.per_device_train_batch_size == ds_config[
            "train_micro_batch_size_per_gpu"]
        assert args.gradient_accumulation_steps == ds_config[
            "train_batch_size"] / ds_config["train_micro_batch_size_per_gpu"]
        try:
            layer_reduction_enabled = ds_config["compression_training"][
                "layer_reduction"]["enabled"]
        except:
            layer_reduction_enabled = False
        if ds_config["compression_training"]["sparse_pruning"]["shared_parameters"]["enabled"] or \
            ds_config["compression_training"]["row_pruning"]["shared_parameters"]["enabled"] or \
                ds_config["compression_training"]["head_pruning"]["shared_parameters"]["enabled"]:
            prune_enabled = True
        else:
            prune_enabled = False
        quantization_enabled = ds_config["compression_training"][
            "weight_quantization"]["shared_parameters"]["enabled"]
        if quantization_enabled:
            assert args.weight_bit == ds_config["compression_training"][
                "weight_quantization"]["different_groups"]["wq1"]["params"][
                    "target_bits"]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.ERROR)
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    def print_rank_0(msg):
        if args.local_rank <= 0:
            print(msg)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    torch.distributed.barrier()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else
                     args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32", "float64"
        ]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    output_mode = output_modes[args.task_name]
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=num_labels,
                                        finetuning_task=args.task_name)
    if layer_reduction_enabled:
        config.num_hidden_layers = ds_config["compression_training"][
            "layer_reduction"][
                'keep_number_layer']  #<==========================================here we assume there is an "num_hidden_layers" argument
    origin_num_attention_heads = config.num_hidden_layers
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    #### load teacher models
    if args.distill_method != 'zero_stage':
        if not args.model_name_or_path_teacher:
            args.model_name_or_path_teacher = args.model_name_or_path
        teacher_config = AutoConfig.from_pretrained(
            args.model_name_or_path_teacher,
            num_labels=num_labels,
            finetuning_task=args.task_name)
        teacher_model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path_teacher,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=teacher_config,
        )
        teacher_model.to(device)
        if args.pretrained_dir_teacher is not None:
            teacher_model.load_state_dict(
                torch.load(args.pretrained_dir_teacher))
    # model inititalization, config,
    model.to(device)
    if layer_reduction_enabled:
        student_initialization(model, teacher_model, args.deepspeed_config)
    else:
        if args.pretrained_dir_student is not None:
            model.load_state_dict(torch.load(
                args.pretrained_dir_student))  # pre-trained checkpoint

    # add more parameters to the model, which does not exist in the pre-trained model
    if quantization_enabled or prune_enabled:
        model = compress(model, args.deepspeed_config)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names
            if name != "label"
        ]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    def replace_config(model_tmp, label_to_id=None):
        if (model_tmp.config.label2id !=
                PretrainedConfig(num_labels=num_labels).label2id
                and args.task_name is not None and not is_regression):
            # Some have all caps in their config, some don't.
            label_name_to_id = {
                k.lower(): v
                for k, v in model_tmp.config.label2id.items()
            }
            if list(sorted(label_name_to_id.keys())) == list(
                    sorted(label_list)):
                logger.info(
                    f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                    "Using it!")
                label_to_id = {
                    i: label_name_to_id[label_list[i]]
                    for i in range(num_labels)
                }
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif args.task_name is None:
            label_to_id = {v: i for i, v in enumerate(label_list)}
        if label_to_id is not None:
            model_tmp.config.label2id = label_to_id
            model_tmp.config.id2label = {
                id: label
                for label, id in config.label2id.items()
            }
        elif args.task_name is not None and not is_regression:
            model_tmp.config.label2id = {
                l: i
                for i, l in enumerate(label_list)
            }
            model_tmp.config.id2label = {
                id: label
                for label, id in config.label2id.items()
            }

    label_to_id = None
    replace_config(model, label_to_id=label_to_id)

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = ((examples[sentence1_key], ) if sentence2_key is None else
                 (examples[sentence1_key], examples[sentence2_key]))
        result = tokenizer(*texts,
                           padding=padding,
                           max_length=args.max_length,
                           truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name ==
                                      "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        mm_eval_dataset = processed_datasets["validation_mismatched"]
        mm_eval_sampler = SequentialSampler(mm_eval_dataset)
        mm_eval_dataloader = DataLoader(
            mm_eval_dataset,
            collate_fn=default_data_collator,
            sampler=mm_eval_sampler,
            batch_size=args.per_device_eval_batch_size)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps /
                                          num_update_steps_per_epoch)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # Prepare the model first
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    if args.distill_method != 'zero_stage':
        teacher_model, _, _, _ = deepspeed.initialize(args=args,
                                                      model=teacher_model)
    # Train!
    def to_device(batch):
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
        return output

    def eval(model):
        # Get the metric function
        if args.task_name is not None:
            metric = load_metric("glue", args.task_name)
        else:
            metric = load_metric("accuracy")

        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(
                dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )
        eval_metric = metric.compute()
        eval_metric1 = None
        if args.task_name == 'mnli':
            metric1 = load_metric("accuracy")
            for step, batch in enumerate(mm_eval_dataloader):
                batch = to_device(batch)
                outputs = model(**batch)
                predictions = outputs.logits.argmax(
                    dim=-1) if not is_regression else outputs.logits.squeeze()
                metric1.add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )
            eval_metric1 = metric1.compute()
        return eval_metric, eval_metric1

    def arrange_output(task_name, results, previous_best, best_dev_acc):
        result = results[0]
        result1 = results[1]
        save_model = False
        if task_name in acc_tasks:
            if task_name in ['sst2', 'qnli', 'rte']:
                current_result = f"acc:{result['accuracy']}"
            elif task_name == 'mnli':
                current_result = f"acc/mm-acc:{result['accuracy']}/{result1['accuracy']}"
            elif task_name in ['mrpc', 'qqp']:
                current_result = f"f1/acc:{result['f1']}/{result['accuracy']}"
            if result['accuracy'] > best_dev_acc:
                save_model = True
                best_dev_acc = result['accuracy']
                previous_best = current_result

        elif task_name in corr_tasks:
            current_result = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']}"
            if result['corr'] > best_dev_acc:
                best_dev_acc = result['corr']
                save_model = True
                previous_best = current_result
        elif task_name in mcc_tasks:
            current_result = f"mcc:{result['matthews_correlation']}"
            if result['matthews_correlation'] > best_dev_acc:
                best_dev_acc = result['matthews_correlation']
                save_model = True
                previous_best = current_result
        return current_result, previous_best, best_dev_acc, save_model

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    previous_best = None
    args.eval_step = default_params[args.task_name]["eval_step"]
    #<===========================================================================eval
    if args.distill_method != 'zero_stage':
        teacher_out = eval(teacher_model)
        teacher_result, _, best_dev_acc, _ = arrange_output(
            args.task_name, teacher_out, previous_best, 0)
        print_rank_0(f"teacher model: {teacher_result}")
    else:
        teacher_result = None
    best_dev_acc = 0.0
    stat_history = {
        "lr1": [],
        "lr2": [],
        'train_att_loss': [],
        'train_ffn_loss': [],
        'train_loss': [],
        'eval': [],
    }

    def train(model,
              previous_best=None,
              best_dev_acc=0,
              text_note='',
              completed_steps=0,
              pred_distill=False,
              intermediate_distill=False):
        loss_mse = MSELoss()
        print_rank_0("***** Running training *****")
        out = eval(model)
        current_result, _, _, _ = arrange_output(args.task_name, out,
                                                 previous_best, best_dev_acc)
        print_rank_0(
            f"at step 0 the (student) model's performance for {args.task_name}: {current_result}"
        )
        tr_loss, tr_rep_loss, tr_cls_loss, tr_att_loss = 0., 0., 0., 0.,
        for epoch in range(args.num_train_epochs):
            model.train()
            start_time = time.time()
            for step, batch in enumerate(train_dataloader):
                batch = to_device(batch)
                outputs = model(**batch,
                                output_attentions=True,
                                output_hidden_states=True)
                student_logits, student_reps, student_atts = outputs.logits, outputs.hidden_states, outputs.attentions

                att_loss, rep_loss, cls_loss, loss = 0., 0., 0., 0.
                if not pred_distill and not intermediate_distill:
                    loss += outputs.loss
                else:
                    with torch.no_grad():
                        outputs_teacher = teacher_model(
                            **batch,
                            output_attentions=True,
                            output_hidden_states=True)
                    teacher_logits, teacher_reps, teacher_atts = outputs_teacher.logits, outputs_teacher.hidden_states, outputs_teacher.attentions

                if pred_distill:
                    if output_mode == "classification":
                        cls_loss = soft_cross_entropy(student_logits,
                                                      teacher_logits)
                    elif output_mode == "regression":
                        cls_loss = loss_mse(student_logits, teacher_logits)
                    loss += cls_loss
                    tr_cls_loss += cls_loss.item()

                if intermediate_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    if layer_reduction_enabled:
                        teacher_layers = [
                            x for x in ds_config["compression_training"]
                            ["layer_reduction"]['teacher_layer']
                        ]
                        att_list = [x for x in teacher_layers]
                        rep_list = [
                            teacher_layers[0] - 1,
                        ] + [x + 1 for x in teacher_layers]
                    else:
                        layers_per_block = int(
                            teacher_layer_num /
                            student_layer_num)  #2###[1, 3, 5, 7, 9, 11]
                        att_list = [
                            i * layers_per_block + layers_per_block - 1
                            for i in range(student_layer_num)
                        ]
                        rep_list = [
                            i * layers_per_block
                            for i in range(student_layer_num + 1)
                        ]  ###[0, 2, 4, 6, 8, 10, 12]

                    if completed_steps % 1000 == 0:
                        print_rank_0(
                            f"check at step:{completed_steps}, teacher_rep_list_{teacher_layer_num}: {rep_list}, teacher_att_list_{student_layer_num}:{att_list}"
                        )

                    new_teacher_reps = [teacher_reps[i] for i in rep_list]
                    new_teacher_atts = [teacher_atts[i] for i in att_list]
                    for student_att, teacher_att in zip(
                            student_atts, new_teacher_atts):
                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss
                        tr_att_loss += att_loss.item()

                    for student_rep, teacher_rep in zip(
                            student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss
                    tr_rep_loss += rep_loss.item()
                    loss += rep_loss + att_loss
                    tr_loss += loss.item()

                model.backward(
                    loss
                )  #<=======================when using deepspedd engine fp16, we should not use loss.backward()
                model.step()
                if step % args.gradient_accumulation_steps == 0 or step == len(
                        train_dataloader) - 1:
                    completed_steps += 1
                    optimizer.zero_grad()

                if completed_steps % args.eval_step == 0 and completed_steps >= 10:
                    loss = tr_loss / (step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)
                    print_rank_0(
                        f"***** Running evaluation Stage {text_note}*****")
                    print_rank_0("  {} step of {}".format(
                        completed_steps, args.max_train_steps))
                    model.eval()
                    result = eval(model)
                    model.train()
                    stat_history['lr1'].append(optimizer.param_groups[0]["lr"])
                    stat_history['lr2'].append(optimizer.param_groups[0]["lr"])
                    try:
                        stat_history['train_ffn_loss'].append(rep_loss.item())
                    except:
                        stat_history['train_ffn_loss'].append(rep_loss)
                    try:
                        stat_history['train_att_loss'].append(att_loss.item())
                    except:
                        stat_history['train_att_loss'].append(att_loss)
                    try:
                        stat_history['train_loss'].append(loss.item())
                    except:
                        stat_history['train_loss'].append(loss)
                    try:
                        stat_history['eval'].append(result)
                    except:
                        stat_history['eval'].append({-1})

                    current_result, previous_best, best_dev_acc, save_model = arrange_output(
                        args.task_name, result, previous_best, best_dev_acc)
                    try:
                        print_rank_0(
                            '{' +
                            f"eval_result: {current_result}, step: {completed_steps/args.max_train_steps}, train_loss: {stat_history['train_loss'][-1]}, train_ffn_loss: {stat_history['train_ffn_loss'][-1]},  train_att_loss:{stat_history['train_att_loss'][-1]}, lr1: { stat_history['lr1'][-1]}, lr2: { stat_history['lr2'][-1]}, "
                            + '}')
                    except:
                        print_rank_0(current_result)
                    if previous_best is not None:
                        print_rank_0(
                            f"teacher_result: {teacher_result}\nPrevious best = {previous_best}"
                        )

                    if save_model and args.save_best_checkpoint:
                        print_rank_0(
                            f'new best checkpoint, saving model to {args.output_dir}'
                        )
                        print_rank_0(
                            f"Task {args.task_name}, teacher_result: {teacher_result}\n {previous_best}"
                        )
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        quant_model = copy.deepcopy(model_to_save)
                        if not ds_config["fp16"]["enabled"]:
                            for name, module in quant_model.named_modules():
                                if hasattr(module, 'weight_quantizer'):
                                    module.weight.data = module.weight_quantizer(
                                        module.weight, args.weight_bit, None,
                                        None,
                                        module.weight_quantize_num_groups)
                                else:
                                    pass
                        WEIGHTS_NAME = "pytorch_model.bin"
                        CONFIG_NAME = 'config.json'
                        output_dir = os.path.join(args.output_dir, 'best')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_model_file = os.path.join(
                            output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(
                            output_dir, CONFIG_NAME)
                        torch.save(quant_model.state_dict(), output_model_file)
                        if args.local_rank in [-1, 0]:
                            if prune_enabled:
                                if ds_config["compression_training"][
                                        "head_pruning"]["shared_parameters"][
                                            "enabled"]:
                                    config.num_attention_heads = int(
                                        origin_num_attention_heads *
                                        ds_config["compression_training"]
                                        ['head_pruning']["different_groups"]
                                        ["rp1"]["params"]["dense_ratio"])
                            model_to_save.config.to_json_file(
                                output_config_file)
                            tokenizer.save_vocabulary(output_dir)
                        if args.deepspeed:
                            new_json_path = os.path.join(
                                output_dir, "ds_config.json")
                            with open(new_json_path, 'w') as f:
                                json.dump(ds_config, f)
                    tr_loss, tr_rep_loss, tr_cls_loss, tr_att_loss = 0., 0., 0., 0.,
                if completed_steps >= args.max_train_steps:
                    break
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print_rank_0(
                f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')

        stat_history_path = os.path.join(args.output_dir, 'stat_history.pt')
        torch.save(stat_history, stat_history_path)
        model.eval()
        result = eval(model)
        current_result, previous_best, best_dev_acc, _ = arrange_output(
            args.task_name, result, previous_best, best_dev_acc)
        print_rank_0(
            f"Teacher perforamnce = {teacher_result} \n Previous best = {previous_best}"
        )
        print_rank_0(f"Finish training. Final accuracy is {current_result}")
        return previous_best, best_dev_acc, completed_steps, model

    previous_best = 0
    best_dev_acc = 0
    completed_steps = 0
    if args.distill_method == 'zero_stage':
        previous_best, best_dev_acc, completed_steps, model_engine = train(
            model_engine,
            previous_best,
            best_dev_acc,
            '0/0',
            completed_steps,
            pred_distill=False,
            intermediate_distill=False)
    elif args.distill_method == 'one_stage':
        previous_best, best_dev_acc, completed_steps, model_engine = train(
            model_engine,
            previous_best,
            best_dev_acc,
            '1/1',
            completed_steps,
            pred_distill=True,
            intermediate_distill=True)

    if args.save_last_model:
        output_dir = os.path.join(args.output_dir, 'final')
        if args.clean_last_model:
            # try:
            model_engine = redundant_clean(model_engine, args.deepspeed_config)
            output_dir = os.path.join(args.output_dir, 'final_clean')
            # except:
            #     print_rank_0 ("not implemented")
            #     pass
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if prune_enabled:
            for module in model_engine.modules():
                if hasattr(module, 'num_attention_heads'):
                    ratio = ds_config["compression_training"]['head_pruning'][
                        "different_groups"]["rp1"]["params"]["dense_ratio"]
                    config.num_attention_heads = int(
                        origin_num_attention_heads * ratio)
                    module.num_attention_heads = int(
                        module.num_attention_heads * ratio)
                    module.all_head_size = int(module.num_attention_heads * 64)
        model_engine.eval()
        result = eval(model_engine)
        current_result, previous_best, best_dev_acc, _ = arrange_output(
            args.task_name, result, previous_best, best_dev_acc)
        print_rank_0(
            f"Clean last_iter models, and the accuracy of the clean last_iter model is {current_result}"
        )

        model_to_save = model_engine.module if hasattr(
            model_engine, 'module') else model_engine
        model_to_save = copy.deepcopy(model_to_save)
        WEIGHTS_NAME = "pytorch_model.bin"
        CONFIG_NAME = 'config.json'
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        if args.local_rank in [-1, 0]:
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir)
        if args.deepspeed:
            new_json_path = os.path.join(args.output_dir, "ds_config.json")
            with open(new_json_path, 'w') as f:
                json.dump(ds_config, f)


if __name__ == "__main__":
    main()
