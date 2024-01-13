import argparse
from pathlib import Path
from postprocess_results import get_summary, ResponseDetails
import json
from datetime import datetime
from dataclasses import asdict

# For these arguments, users can provide multiple values when running the
# benchmark. The benchmark will iterate over all possible combinations.
SERVER_PARAMS = ["tp_size", "max_ragged_batch_size", "replica_num"]
CLIENT_PARAMS = [
    "mean_prompt_length",
    "mean_max_new_tokens",
    "num_clients",
    "num_requests",
]


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
        "--cmd", type=str, choices=["start", "stop", "restart"], default="start"
    )

    # Client args
    client_parser = argparse.ArgumentParser(add_help=False)
    client_parser.add_argument(
        "--mean_prompt_length", type=int, nargs="+", default=[2600]
    )
    client_parser.add_argument(
        "--mean_max_new_tokens", type=int, nargs="+", default=[60]
    )
    client_parser.add_argument("--num_clients", type=int, nargs="+", default=[2])
    client_parser.add_argument("--num_requests", type=int, nargs="+", default=[512])
    client_parser.add_argument("--max_prompt_length", type=int, default=4000)
    client_parser.add_argument("--prompt_length_var", type=float, default=0.3)
    client_parser.add_argument("--max_new_tokens_var", type=float, default=0.3)
    client_parser.add_argument("--warmup", type=int, default=1)
    client_parser.add_argument("--use_thread", action="store_true")
    client_parser.add_argument("--stream", action="store_true")
    client_parser.add_argument("--out_json_path", type=Path, default=None)

    # Create the parser, inheriting from the server and/or client parsers
    parents = []
    if server_args:
        parents.append(server_parser)
    if client_args:
        parents.append(client_parser)

    # Common args
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--deployment_name", type=str, default=None)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--use_defaults", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    if server_args and not client_args:
        # If running server, make sure only single values were passed for parameters
        for param in SERVER_PARAMS:
            if len(getattr(args, param)) > 1:
                raise ValueError(
                    f"Cannot specify multiple values for {param} when running server"
                )
            setattr(args, param, getattr(args, param)[0])

    if client_args and not server_args:
        # If running client, make sure only single values were passed for parameters
        for param in CLIENT_PARAMS:
            if len(getattr(args, param)) > 1:
                raise ValueError(
                    f"Cannot specify multiple values for {param} when running client"
                )
            setattr(args, param, getattr(args, param)[0])

    if not (client_args and server_args):
        # Generate deployment name if not provided
        if args.deployment_name is None:
            args.deployment_name = get_deployment_name(
                model=args.model,
                tp_size=args.tp_size,
                max_ragged_batch_size=args.max_ragged_batch_size,
            )

    return args


def get_deployment_name(
    model: str, tp_size: int, max_ragged_batch_size: int, num_replicas: int
) -> str:
    return f"{model}-tp{tp_size}-b{max_ragged_batch_size}-r{num_replicas}"


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

    if args.out_json_path is not None:
        with open(args.out_json_path, "w") as f:
            args_dict["out_json_path"] = str(
                args.out_json_path
            )  # Path is not JSON serializable
            data = {
                "args": args_dict,
                "time": str(datetime.now()),
                "response_details": [asdict(r) for r in response_details],
            }
            json.dump(data, f, indent=2)
