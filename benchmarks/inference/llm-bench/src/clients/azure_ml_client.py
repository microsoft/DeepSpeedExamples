from .base import BaseClient
from ..config import BaseConfigModel
from ..status import Status
from ..prompt import Prompt
from ..response import Response

import requests
import json
from typing import Any, Dict, Optional


class AzureMLClientConfig(BaseConfigModel):
    api_url: str = ""
    api_key: str = ""
    deployment_name: str = ""


class AzureMLClient(BaseClient):
    def __init__(self, config: AzureMLClientConfig) -> None:
        super().__init__(config)
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.deployment_name = config.deployment_name

    def start_service(self) -> Status:
        pass

    def stop_service(self) -> Status:
        pass

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        if prompt.streaming:
            raise ValueError("AzureMLClient does not support streaming prompts.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
            "azureml-model-deployment": self.deployment_name,
        }
        pload = {
            "input_data": {
                "input_string": [
                    prompt.text,
                ],
                "parameters": {
                    "max_tokens": prompt.max_new_tokens,
                    "return_full_text": prompt.return_full_text,
                },
            }
        }
        return {"url": self.api_url, "headers": headers, "json": pload, "timeout": 180}

    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        while True:
            try:  # Sometimes the AML endpoint will return an error, so we send the request again
                response = requests.post(**request_kwargs)
                output = json.loads(response.content)
                break
            except Exception as e:
                print(f"Connection failed with {e}. Retrying AML request")

        return output

    def process_response(self, raw_response: Any) -> Response:
        response_text = raw_response[0]
        return Response(response_text)
