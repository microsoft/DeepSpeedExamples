from abc import ABC, abstractmethod

from ..prompt import Prompt
from ..response import Response
from ..status import Status
from ..config import BaseConfigModel

from typing import Any, Dict

class BaseClient(ABC):
    def __init__(self, config:BaseConfigModel):
        self.config = config

    @abstractmethod
    def start_service(self) -> Status:
        pass

    @abstractmethod
    def stop_service(self) -> Status:
        pass

    @abstractmethod
    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        pass

    @abstractmethod
    def send_request(self, request_kwargs: Dict[str,Any]) -> Any:
        pass

    @abstractmethod
    def process_response(self, raw_response: Any) -> Response:
        pass