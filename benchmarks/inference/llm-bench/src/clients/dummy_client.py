from abc import ABC, abstractmethod

from ..config import BaseConfigModel
from ..prompt import Prompt
from ..response import Response
from ..status import Status
from .base import BaseClient

from typing import Any, Dict
import time
import random


class DummyClientConfig(BaseConfigModel):
    pass


class DummyClient(BaseClient):
    def start_service(self) -> Status:
        return Status("OK")

    def stop_service(self) -> Status:
        return Status("OK")

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        return {"input_text": prompt.text, "max_new_tokens": prompt.max_new_tokens}

    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        time.sleep(random.uniform(1, 2))
        return request_kwargs["input_text"] * 2

    def process_response(self, raw_response: Any) -> Response:
        return Response(raw_response)
