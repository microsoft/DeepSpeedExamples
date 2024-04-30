from abc import ABC, abstractmethod

from ..prompt import Prompt
from ..response import Response
from ..status import Status
from ..config import BaseConfigModel

from typing import Any, Dict

class BaseClient(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def start_service(config: BaseConfigModel) -> Status:
        pass

    @staticmethod
    @abstractmethod
    def stop_service(config: BaseConfigModel) -> Status:
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