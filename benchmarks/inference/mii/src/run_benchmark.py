# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from client import run_client
from model_defaults import MODEL_DEFAULTS
from server import start_server, stop_server
from utils import (
    parse_args,
    print_summary,
    save_json_results,
    get_args_product,
    SERVER_PARAMS,
    CLIENT_PARAMS,
)


def run_benchmark() -> None:
    args = parse_args(server_args=True, client_args=True)

    if not args.no_model_defaults:
        if args.model not in MODEL_DEFAULTS:
            raise ValueError(
                f"Model {args.model} not in MODEL_DEFAULTS. "
                f"Please specify arguments manually and use the --no_model_defaults flag."
            )
        for k, v in MODEL_DEFAULTS[args.model].items():
            setattr(args, k, v)

    for server_args in get_args_product(args, which=SERVER_PARAMS):
        start_server(server_args)

        for client_args in get_args_product(server_args, which=CLIENT_PARAMS):
            response_details = run_client(client_args)
            print_summary(client_args, response_details)
            save_json_results(client_args, response_details)

        stop_server(server_args)


if __name__ == "__main__":
    run_benchmark()
