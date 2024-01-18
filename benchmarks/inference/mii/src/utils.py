import argparse
from pathlib import Path
from postprocess_results import get_summary, ResponseDetails
import json
from datetime import datetime
from dataclasses import asdict
import itertools
from typing import Iterator, List
import copy
import os

# For these arguments, users can provide multiple values when running the
# benchmark. The benchmark will iterate over all possible combinations.
SERVER_PARAMS = ["tp_size", "max_ragged_batch_size", "num_replicas"]
CLIENT_PARAMS = ["mean_prompt_length", "mean_max_new_tokens", "num_clients"]


def parse_args(
    server_args: bool = False, client_args: bool = False
) -> argparse.Namespace:
    if not (server_args or client_args):
        raise ValueError("Must specify server_args or client_args or both")

    # Server args
    server_parser = argparse.ArgumentParser(add_help=False)
    server_parser.add_argument("--tp_size", type=int, nargs="+", default=[1])
    server_parser.add_argument(
        "--max_ragged_batch_size", type=int, nargs="+", default=[768]
    )
    server_parser.add_argument("--num_replicas", type=int, nargs="+", default=[1])
    server_parser.add_argument(
        "cmd", type=str, nargs="?", choices=["start", "stop", "restart"]
    )

    # Client args
    client_parser = argparse.ArgumentParser(add_help=False)
    client_parser.add_argument(
        "--mean_prompt_length", type=int, nargs="+", default=[2600]
    )
    client_parser.add_argument(
        "--mean_max_new_tokens", type=int, nargs="+", default=[60]
    )
    client_parser.add_argument(
        "--num_clients",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    )
    client_parser.add_argument("--num_requests", type=int, default=512)
    client_parser.add_argument("--max_prompt_length", type=int, default=4000)
    client_parser.add_argument("--prompt_length_var", type=float, default=0.3)
    client_parser.add_argument("--max_new_tokens_var", type=float, default=0.3)
    client_parser.add_argument("--warmup", type=int, default=1)
    client_parser.add_argument("--use_thread", action="store_true")
    client_parser.add_argument("--stream", action="store_true")
    client_parser.add_argument("--out_json_dir", type=Path, default="./results/")

    # Create the parser, inheriting from the server and/or client parsers
    parents = []
    if server_args:
        parents.append(server_parser)
    if client_args:
        parents.append(client_parser)

    # Common args
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--deployment_name", type=str, default="mii-benchmark-deployment"
    )
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--no_model_defaults", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    return args


def get_args_product(
    args: argparse.Namespace, which: List[str] = None
) -> Iterator[argparse.Namespace]:
    if which is None:
        return copy.deepcopy(args)
    arg_values_product = itertools.product(*[getattr(args, k) for k in which])
    for arg_values in arg_values_product:
        args_copy = copy.deepcopy(args)
        for k, v in zip(which, arg_values):
            setattr(args_copy, k, v)
        yield args_copy


def get_results_path(args: argparse.Namespace) -> Path:
    if args.vllm:
        lib_path = "vllm"
    else:
        lib_path = "fastgen"
    return Path(
        args.out_json_dir,
        f"{lib_path}/",
        "-".join(
            (
                args.model.replace("/", "_"),
                f"tp{args.tp_size}",
                f"bs{args.max_ragged_batch_size}",
                f"replicas{args.num_replicas}",
                f"prompt{args.mean_prompt_length}",
                f"gen{args.mean_max_new_tokens}",
                f"clients{args.num_clients}",
            )
        )
        + ".json",
    )


def output_summary(args, response_details):
    args_dict = vars(args)
    ps = get_summary(args_dict, response_details)
    print(
        f"Deployment: {args.deployment_name} Clients: {args.num_clients}, "
        + f"Prompt (mean): {args.mean_prompt_length} tokens, "
        + f"Generation (mean): {args.mean_max_new_tokens} tokens, "
        + f"Query throughput: {ps.throughput:.3f} queries/s, "
        + f"Token throughput (total): {ps.tokens_per_sec:.3f} tokens/s, "
        + f"Query latency: {ps.latency:.3f} s, "
        + f"Token generation latency: {ps.token_gen_latency:.3f} s/token, "
        + f"First token received: {ps.first_token_latency:.3f} s"
    )

    out_json_path = get_results_path(args)

    os.makedirs(out_json_path.parent, exist_ok=True)

    with open(out_json_path, "w") as f:
        args_dict["out_json_dir"] = str(out_json_path)  # Path is not JSON serializable
        data = {
            "args": args_dict,
            "time": str(datetime.now()),
            "response_details": [asdict(r) for r in response_details],
        }
        json.dump(data, f, indent=2)
