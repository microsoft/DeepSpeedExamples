from abc import ABC, abstractmethod

from ..config import BaseConfigModel
from ..prompt import Prompt
from ..response import Response
from ..status import Status
from .base import BaseClient

from transformers import AutoTokenizer
from typing import Any, Dict
import time
import random


class DummyClientConfig(BaseConfigModel):
    model: str


class DummyClient(BaseClient):
    def __init__(self, config: DummyClientConfig) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

    def start_service(self) -> Status:
        return Status("OK")

    def stop_service(self) -> Status:
        return Status("OK")

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        return {"input_text": prompt.text, "max_new_tokens": prompt.max_new_tokens}

    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        time.sleep(random.uniform(1, 2))
        output_text = self.tokenizer.decode(
            random.choices(
                self.tokenizer.encode(request_kwargs["input_text"]),
                k=request_kwargs["max_new_tokens"],
            )
        )
        return output_text

    def process_response(self, raw_response: Any) -> str:
        return raw_response
