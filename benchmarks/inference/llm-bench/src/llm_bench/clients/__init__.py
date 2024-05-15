from .azure_ml_client import AzureMLClientConfig, AzureMLClient
from .dummy_client import DummyClientConfig, DummyClient
from .fastgen_client import FastGenClientConfig, FastGenClient
from .vllm_client import vLLMClientConfig, vLLMClient

# TODO: Make these auto-generated, so new clients can be added without changing this file
client_config_classes = {
    "dummy": DummyClientConfig,
    "azure_ml": AzureMLClientConfig,
    "fastgen": FastGenClientConfig,
    "vllm": vLLMClientConfig,
}
client_classes = {
    "dummy": DummyClient,
    "azure_ml": AzureMLClient,
    "fastgen": FastGenClient,
    "vllm": vLLMClient,
}
