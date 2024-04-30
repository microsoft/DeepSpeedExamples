from .base import BaseClient
from ..status import Status
from ..prompt import Prompt
from ..response import Response
from ..config import BaseConfigModel
from pydantic import Field
from typing import Optional, Dict, Any
import time

class FastGenClientConfig(BaseConfigModel):
    model: str = Field(..., description="HuggingFace.co model name")
    deployment_name: str = "fastgen-benchmark-deployment"
    tp_size: int = 1
    num_replicas: int = 1
    max_ragged_batch_size: int = 768
    quantization_mode: Optional[str] = None

class FastGenClient(BaseClient):
    def __init__(self, config: FastGenClientConfig):
        import mii
        self.mii_client = mii.client(config.deployment_name)
        self.streaming = config.streaming

    @staticmethod
    def start_service(config: FastGenClientConfig) -> Status:
        import mii
        from deepspeed.inference import RaggedInferenceEngineConfig, DeepSpeedTPConfig
        from deepspeed.inference.v2.ragged import DSStateManagerConfig

        tp_config = DeepSpeedTPConfig(tp_size=config.tp_size)
        mgr_config = DSStateManagerConfig(
            max_ragged_batch_size=config.max_ragged_batch_size,
            max_ragged_sequence_count=config.max_ragged_batch_size,
        )
        inference_config = RaggedInferenceEngineConfig(
            tensor_parallel=tp_config, state_manager=mgr_config
        )
        mii.serve(
            config.model,
            deployment_name=config.deployment_name,
            tensor_parallel=config.tp_size,
            inference_engine_config=inference_config,
            replica_num=config.num_replicas,
            quantization_mode=config.quantization_mode
        )

    @staticmethod
    def stop_service(config: FastGenClientConfig) -> Status:
        import mii
        mii.client(config.deployment_name).terminate_server()

    def _streaming_callback(self, raw_response) -> None:
        self.streaming_response_tokens.append(raw_response[0].generated_text)
        time_now = time.time()
        self.streaming_token_gen_time.append(time_now - time_last_token)
        time_last_token = time_now

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        request_kwargs = {"prompts": prompt.text, "max_new_tokens": prompt.max_new_tokens}
        if self.streaming:
            self.streaming_response_tokens = []
            self.streaming_token_gen_time = []
            self.streaming_time_last_token = None
            request_kwargs["streaming_fn"] = self._streaming_callback
        return request_kwargs

    def send_request(self, request_kwargs: Dict[str,Any]) -> Any:
        if self.streaming:
            self.streaming_time_last_token = time.time()
        response = self.mii_client(**request_kwargs)
        if self.streaming:
            response = self.streaming_response_tokens
        
        return response

    def process_response(self, raw_response: Any) -> Response:
        if not self.streaming:
            raw_response = raw_response[0].generated_text
        return Response(raw_response)