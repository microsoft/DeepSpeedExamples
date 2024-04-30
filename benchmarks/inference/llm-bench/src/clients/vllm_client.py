from .base import BaseClient
from ..status import Status
from ..prompt import Prompt
from ..response import Response
from ..config import BaseConfigModel
from pydantic import Field
from typing import Optional, Dict, Any
import subprocess
import time
import requests
import json

class vLLMClientConfig(BaseConfigModel):
    model: str = Field(..., description="HuggingFace.co model name")
    tp_size: int = 1
    port: int = 26500

class vLLMClient(BaseClient):
    def __init__(self, config: vLLMClientConfig) -> None:
        pass

    @staticmethod
    def start_service(config: vLLMClientConfig) -> Status:
        vllm_cmd = (
            "python",
            "-m",
            "vllm.entrypoints.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(config.port),
            "--tensor-parallel-size",
            str(config.tp_size),
            "--model",
            config.model,
        )
        p = subprocess.Popen(
            vllm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, close_fds=True
        )
        start_time = time.time()
        timeout_after = 60 * 5  # 5 minutes
        while True:
            line = p.stderr.readline().decode("utf-8")
            if "Application startup complete" in line:
                break
            if "error" in line.lower():
                p.terminate()
                #self.stop_service(config)
                raise RuntimeError(f"Error starting VLLM server: {line}")
            if time.time() - start_time > timeout_after:
                p.terminate()
                #self.stop_service(config)
                raise TimeoutError("Timed out waiting for VLLM server to start")
            time.sleep(0.01)

    @staticmethod
    def stop_service(config: vLLMClientConfig) -> Status:
        vllm_cmd = ("pkill", "-f", "vllm.entrypoints.api_server")
        p = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        api_url = "http://localhost:26500/generate"
        headers = {"User-Agent": "Benchmark Client"}
        pload = {
            "prompt": prompt.text,
            "n": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": prompt.max_new_tokens,
            "ignore_eos": False,
        }
        return {"url": api_url, "headers": headers, "json": pload, "timeout": 180}

    def send_request(self, request_kwargs: Dict[str,Any]) -> Any:
        response = requests.post(**request_kwargs)
        output = json.loads(response.content)
        return output

    def process_response(self, raw_response: Any) -> Response:
        return Response(raw_response["text"])