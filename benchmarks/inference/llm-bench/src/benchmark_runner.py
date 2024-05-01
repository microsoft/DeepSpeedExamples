from .config import BaseConfigModel
from .prompt import PromptGenerator, PromptConfig
from .clients import client_classes
from typing import List, Optional
from pydantic import Field
from pathlib import Path
import multiprocessing
import threading
import queue
import time
import yaml
import itertools
from tqdm import tqdm
from loguru import logger


class BenchmarkConfig(BaseConfigModel):
    model: str = Field(..., description="HuggingFace.co model name")
    api: str = "azure_ml"
    warmup_requests: int = 1
    result_dir: Path = Path("./results")
    use_threading: bool = False
    config_files: List[Path] = []
    num_clients: List[int] = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
    prompt_generator_seed: Optional[int] = None
    num_requests_per_client: int = 16
    max_prompt_length: int = 4000
    prompt_length: List[int] = [2600]
    prompt_length_var: float = 0.3
    max_new_tokens: List[int] = [60]
    max_new_tokens_var: float = 0.3
    streaming: bool = False

class BenchmarkRunner():
    def __init__(self, benchmark_config: BaseConfigModel, client_config: BaseConfigModel) -> None:
        logger.info("Initializing Benchmark Runner")
        self.config = benchmark_config
        self.client_config = client_config
        self.client_class = client_classes[self.config.api]
        self.client_obj = self.client_class(self.client_config)

        self.runnable_cls = multiprocessing.Process
        self.barrier_cls = multiprocessing.Barrier
        self.queue_cls = multiprocessing.Queue
        if self.config.use_threading:
            self.runnable_cls = threading.Thread
            self.barrier_cls = threading.Barrier
            self.queue_cls = queue.Queue

    def _generate_prompts(self, prompt_config: PromptConfig, num_clients: int) -> None:
        logger.info("Generating Prompts")
        prompt_generator = PromptGenerator(prompt_config)
        warmup_prompts = self.config.warmup_requests * num_clients
        workload_prompts = self.config.num_requests_per_client * num_clients
        for prompt in prompt_generator(warmup_prompts + workload_prompts):
            prepared_request = self.client_obj.prepare_request(prompt)
            self.query_queue.put(prepared_request)
        logger.info(f"Generated {warmup_prompts} warmup and {workload_prompts} workload prompts.")

    def _launch_clients(self, num_clients):
        logger.info(f"Launching {num_clients} client(s)")
        self.barrier = self.barrier_cls(num_clients + 1)
        processes = [
            self.runnable_cls(
                target=self._run_client,
                args=(
                    self.barrier,
                    self.query_queue,
                    self.result_queue,
                    self.client_class,
                    self.client_config,
                    self.config.warmup_requests,
                ),
            )
            for _ in range(num_clients)
        ]
        for p in processes:
            p.start()

        total_prompts = num_clients * self.config.num_requests_per_client
        pbar = tqdm(total=total_prompts)

        self.barrier.wait() # Barrier 1 for master process

        num_results = 0
        while num_results != total_prompts:
            num_results = self.result_queue.qsize()
            pbar.update(num_results - pbar.n)
            time.sleep(1)
        pbar.close()

        self.barrier.wait() # Barrier 2 for master process


    @staticmethod
    def _run_client(barrier, query_queue, result_queue, client_class, client_config, warmup_requests):
        client = client_class(client_config)

        for _ in range(warmup_requests):
            request_kwargs = query_queue.get(timeout=1.0)
            _ = client.send_request(request_kwargs)

        barrier.wait() # Barrier 1 for client process
        try:
            while True:
                request_kwargs = query_queue.get(timeout=1.0)
                start_time = time.time()
                raw_response = client.send_request(request_kwargs)
                end_time = time.time()
                request_time = end_time - start_time
                result_queue.put_nowait((raw_response, request_time))
        except queue.Empty:
            pass

        barrier.wait() # Barrier 2 for client process

    def _benchmark_settings(self):
        prompt_config_keys = list(PromptConfig.model_fields.keys()) + ["num_clients"]

        configs_list = []
        for f in self.config.config_files:
            logger.info(f"Generating benchmark run settings from config file: {f}")
            with open(f, "r") as fh:
                file_config = yaml.safe_load(fh)
            for key in prompt_config_keys:
                if key not in file_config:
                    file_config[key] = getattr(self.config, key)
            configs_list.append(file_config)

        if not configs_list:
            logger.info(f"Generating benchmark run settings from command line args")
            configs_list.append({key: getattr(self.config, key) for key in prompt_config_keys})

        all_config_product = []
        for config in configs_list:
            for k, v in config.items():
                if not isinstance(v, list) or isinstance(v, tuple):
                    config[k] = [v]
            for vals in itertools.product(*[config[k] for k in prompt_config_keys]):
                all_config_product.append({k:v for k,v in zip(prompt_config_keys, vals)})

        logger.info(f"Generated {len(all_config_product)} benchmark run setting(s)")

        for config in all_config_product:
            num_clients = config.pop("num_clients")
            prompt_config = PromptConfig(**config)
            yield num_clients, prompt_config

    def _clear_queues(self):
        self.query_queue = self.queue_cls()
        self.result_queue = self.queue_cls()

    def _save_results(self, num_clients, prompt_config):
        response_details = []
        while len(response_details) != num_clients * self.config.num_requests_per_client:
            res = self.result_queue.get()
            # vLLM returns concatinated tokens
            response_details.append(res)
        return response_details

    def run(self):
        self.client_obj.start_service()
        for num_clients, prompt_config in self._benchmark_settings():
            logger.info(f"Running benchmark with {num_clients} client(s) and prompt config: {prompt_config}")
            self._clear_queues()
            self._generate_prompts(prompt_config=prompt_config, num_clients=num_clients)
            #self._prepare_requests()
            self._launch_clients(num_clients=num_clients)
            #self._process_repsonses()
            rd = self._save_results(prompt_config=prompt_config, num_clients=num_clients)
        self.client_obj.stop_service()
        

if __name__ == "__main__":
    from .arg_parsing import parse_args_to_configs
    benchmark_config, client_config = parse_args_to_configs()
    benchmark_runner = BenchmarkRunner(benchmark_config, client_config)
    benchmark_runner.run()