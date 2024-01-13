import argparse
import itertools
from typing import Dict, List

from utils import parse_args, get_deployment_name, output_summary
from server import start_server, stop_server
from client import run_client

MODEL_DEFAULTS = {
    "meta-llama/Llama-2-7b-hf": {
        "max_prompt_length": 4000,
        "mean_prompt_length": (1200, 2600),
        "mean_max_new_tokens": (60, 128),
        "tp_size": (1,),
        "num_clients": (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32),
    },
    "meta-llama/Llama-13b-hf": {
        "max_prompt_length": 4000,
        "mean_prompt_length": (1200, 2600),
        "mean_max_new_tokens": (60, 128),
        "tp_size": (1, 2, 4),
        "num_clients": (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32),
    },
    "meta-llama/Llama-2-70b-hf": {
        "max_prompt_length": 4000,
        "mean_prompt_length": (1200, 2600),
        "mean_max_new_tokens": (60, 128),
        "tp_size": (4, 8),
        "num_clients": (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32),
    },
    "tiiuae/falcon-180B": {
        "max_prompt_length": 2000,
        "mean_prompt_length": (1200, 1900),
        "mean_max_new_tokens": (60, 128),
        "tp_size": (8,),
        "num_clients": (1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32),
    },
}


def get_args_product(args: argparse.Namespace, which: List[str] = None) -> List[Dict]:
    if which is None:
        which = list(vars(args).keys())
    arg_values_product = itertools.product(*[getattr(args, k) for k in which])
    return [
        {k: v for k, v in zip(which, arg_values)} for arg_values in arg_values_product
    ]


def run_benchmark() -> None:
    args = parse_args(server_args=True, client_args=True)

    if args.use_defaults:
        for k, v in MODEL_DEFAULTS[args.model].items():
            setattr(args, k, v)

    # Args to enumerate over for benchmarks
    server_arg_names = ["tp_size", "max_ragged_batch_size", "num_replicas"]
    client_arg_names = [
        "mean_prompt_length",
        "mean_max_new_tokens",
        "num_clients",
        "num_requests",
    ]

    # Run MII benchmarks
    for server_args in get_args_product(args, which=server_arg_names):
        if args.deployment_name is None:
            args.deployment_name = get_deployment_name(model=args.model, **server_args)
        start_server(
            model=args.model,
            deployment_name=args.deployment_name,
            vllm=args.vllm,
            **server_args,
        )

        for client_args in get_args_product(args, which=client_arg_names):
            response_details = run_client(
                model=args.model,
                deployment_name=args.deployment_name,
                max_prompt_length=args.max_prompt_length,
                prompt_length_var=args.prompt_length_var,
                max_new_tokens_var=args.max_new_tokens_var,
                warmup=args.warmup,
                use_thread=args.use_thread,
                stream=args.stream,
                vllm=args.vllm,
                **client_args,
            )
            output_summary(args, response_details)

        stop_server(deployment_name=args.deployment_name, vllm=args.vllm)


if __name__ == "__main__":
    run_benchmark()
