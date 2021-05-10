"""A collection of utility methods for working with the GLUE dataset
Primarily includes methods to:
    - Download raw GLUE data
    - Process GLUE data with a given tokenizer
Can also be run as a script in which case it will download and process the
GLUE data for a specified task, and use a specified tokenizer to process
the data, which is then written to provided output directory.
"""
import argparse
import os
import logging
from typing import Any, Union, Dict, Callable
from datasets import load_dataset, load_metric
from datasets import DatasetDict, Dataset, Metric  # used for typing
from torch.utils.data.dataset import Dataset
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
)


logger = logging.getLogger(__name__)

# specific mapping from glue task to dataset column names
task_columns = {
    "cola": ("sentence",),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# mnli-mm is a special name used by huggingface
actual_task = lambda task: "mnli" if task == "mnli-mm" else task


def num_labels_from_task(task: str) -> int:
    """Return the number of labels for the GLUE task."""
    if task.startswith("mnli"):
        return 3
    elif task.startswith("stsb"):
        return 1
    else:
        # all other glue tasks have 2 class labels
        return 2


def load_metric_from_task(task: str) -> Metric:
    """Load the metric for the corresponding GLUE task."""
    metric = load_metric("glue", actual_task(task))
    return metric


def get_metric_name_from_task(task: str) -> str:
    """Get the name of the metric for the corresponding GLUE task.
    If using `load_best_model_at_end=True` in TrainingArguments then you need
    `metric_for_best_model=metric_name`. Use this method to get the metric_name
    for the corresponding GLUE task.
    """
    if task == "stsb":
        return "pearson"
    elif task == "cola":
        return "matthews_correlation"
    else:
        return "accuracy"


def construct_tokenizer_function(
    tokenizer: PreTrainedTokenizerBase, task: str
) -> Callable[[Union[Dict, Any]], Union[Dict, Any]]:
    """Construct function used to tokenize GLUE data.
    Some GLUE tasks (CoLA and SST2) have single sentence input, while the rest
    have sentence pairs. This method returns a method that applies the appropriate
    tokenizer to an example input based on that tasks sentence_keys.
    Args:
        tokenizer: A Transformers Tokenizer used to convert raw sentences into
            something our model can understand.
        task: Names of the GLUE task.
    Returns:
        A function that applies our tokenizer to example sentence(s) from the
        associated GLUE task.
    """

    sentence_keys = task_columns.get(task)

    if len(sentence_keys) == 1:
        sentence1_key = sentence_keys[0]

        def tokenize_single_sentence(examples: Union[Dict, Any]) -> Union[Dict, Any]:
            return tokenizer(examples[sentence1_key], truncation=True)

        return tokenize_single_sentence

    else:
        sentence1_key, sentence2_key = sentence_keys

        def tokenize_sentence_pair(examples: Union[Dict, Any]) -> Union[Dict, Any]:
            return tokenizer(
                examples[sentence1_key], examples[sentence2_key], truncation=True
            )

        return tokenize_sentence_pair


def load_raw_glue_dataset(task: str) -> Union[DatasetDict, Dataset]:
    dataset = load_dataset("glue", actual_task(task))
    return dataset


def load_encoded_glue_dataset(
    task: str,
    tokenizer: PreTrainedTokenizerBase,
) -> Union[DatasetDict, Dataset]:
    """Load GLUE data, apply tokenizer and split into train/validation."""
    tokenizer_func = construct_tokenizer_function(tokenizer=tokenizer, task=task)
    raw_dataset = load_raw_glue_dataset(task)
    encoded_dataset = raw_dataset.map(tokenizer_func, batched=True)

    validation_key = (
        "validation_mismatched"
        if task == "mnli-mm"
        else "validation_matched"
        if task == "mnli"
        else "validation"
    )

    return encoded_dataset["train"], encoded_dataset[validation_key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="distilbert-base-uncased")
    parser.add_argument("--task", help="Name of GLUE task")
    parser.add_argument(
        "--use_fast",
        action="store_false",
        help="Bool that determines to use fast tokenizer or not. Default is True.",
    )
    parser.add_argument(
        "--output_dir", help="Directory to store tokenized GLUE dataset."
    )
    args, unparsed = parser.parse_known_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_checkpoint, use_fast=args.use_fast
    )

    logger.info("Downloading raw")
    tokenized_dataset = load_encoded_glue_dataset(
        task=args.task.lower(), tokenizer=tokenizer
    )

    logger.info(f"Saving processed dataset to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(args.output_dir)
    logger.info("Done!")
