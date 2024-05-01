from typing import Iterable, Optional
from .config import BaseConfigModel
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Prompt:
    text: str
    num_prompt_tokens: int
    max_new_tokens: int
    streaming: bool = False
    return_full_text: bool = False


class PromptConfig(BaseConfigModel):
    model: str
    max_prompt_length: int
    prompt_length: int
    prompt_length_var: float
    max_new_tokens: int
    max_new_tokens_var: float
    streaming: bool


class PromptGenerator:
    def __init__(self, config: PromptConfig) -> None:
        self.model = config.model
        self.max_prompt_length = config.max_prompt_length
        self.prompt_length = config.prompt_length
        self.prompt_length_var = config.prompt_length_var
        self.max_new_tokens = config.max_new_tokens
        self.max_new_tokens_var = config.max_new_tokens_var
        # TODO: Make this better
        from .sample_input import all_text

        self.input_text = all_text
        self._load_tokenizer()

    def _load_input_text(self) -> None:
        pass

    def _load_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def __call__(self, num_prompts: Optional[int] = None) -> Iterable[Prompt]:
        tokenized_input = self.tokenizer.batch_encode_plus(
            [self.input_text], return_tensors="pt", padding=False
        )["input_ids"][0]

        if num_prompts is None:
            num_prompts = self.config.num_prompts

        for i in range(num_prompts):
            prompt_length = min(
                int(np.random.normal(self.prompt_length, self.prompt_length_var)),
                self.max_prompt_length,
            )
            max_new_tokens = int(
                np.random.normal(self.max_new_tokens, self.max_new_tokens_var)
            )
            yield Prompt(
                text=self.tokenizer.decode(tokenized_input[i : prompt_length + i]),
                num_prompt_tokens=prompt_length,
                max_new_tokens=max_new_tokens,
            )
