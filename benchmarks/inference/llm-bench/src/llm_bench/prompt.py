import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

from .config import BaseConfigModel

# Avoids a warning from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Prompt:
    text: str
    num_tokens: int
    max_new_tokens: int
    streaming: bool = False
    return_full_text: bool = False
    request_kwargs: dict = None


class PromptConfig(BaseConfigModel):
    # TODO: Add descriptions for each field
    model: str
    max_prompt_length: int
    prompt_length: int
    prompt_length_var: float
    max_new_tokens: int
    max_new_tokens_var: float
    streaming: bool


class PromptGenerator:
    def __init__(self, model: str, prompt_text_source: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if os.path.isfile(prompt_text_source):
            with open(prompt_text_source, "r") as f:
                prompt_text_source = f.read()
        self.input_text = prompt_text_source

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def __call__(self, config: PromptConfig, num_prompts: int) -> Iterable[Prompt]:
        tokenized_input = self.tokenizer.batch_encode_plus(
            [self.input_text], return_tensors="pt", padding=False
        )["input_ids"][0]

        # TODO: Add support for prompts longer than source text

        for i in range(num_prompts):
            prompt_length = min(
                int(np.random.normal(config.prompt_length, config.prompt_length_var)),
                config.max_prompt_length,
            )
            max_new_tokens = int(
                np.random.normal(config.max_new_tokens, config.max_new_tokens_var)
            )
            yield Prompt(
                text=self.tokenizer.decode(tokenized_input[i : prompt_length + i]),
                num_tokens=prompt_length,
                max_new_tokens=max_new_tokens,
            )
