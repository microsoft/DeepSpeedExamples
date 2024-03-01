# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import glob
import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools

from postprocess_results import read_json, get_token_latency, get_result_sets

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default=".")
    parser.add_argument(
        "--out_dir", type=Path, default="./plots/percentile_token_latency"
    )
    parser.add_argument("--backend", type=str, choices=["aml", "fastgen", "vllm"], default=["aml", "fastgen", "vllm"], \
                        nargs="+", help="Specify the backends to generate plots for")
    parser.add_argument("--skip_head_token_num", type=int, default=1, help="Specify number of head tokens to skip")
    parser.add_argument("--skip_request_num", type=int, default=1, help="Specify number of requests to skip")
    args = parser.parse_args()
    return args


def extract_values(args, file_pattern):
    files = glob.glob(file_pattern)

    print(f"Found {len(files)}")
    print("\n".join(files))

    latencies = {}
    for f in files:
        prof_args, response_details = read_json(f)
        client_num = prof_args["num_clients"]

        response_details.sort(key=lambda r: r.start_time)

        response_details = response_details[args.skip_request_num:-args.skip_request_num]
        token_latencies = [
            r.token_gen_time[args.skip_head_token_num:-1] for r in response_details
        ]
        flat_latency_list = list(itertools.chain(*token_latencies))
        latencies[client_num] = flat_latency_list
    return latencies


def output_charts(args, model, tp_size, bs, replicas, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    result_file_pattern = f"{model}-tp{tp_size}-bs{bs}-replicas{replicas}-prompt{prompt}-gen{gen}-clients*.json"

    if "vllm" in args.backend:
        vllm_file_pattern = f"{log_dir}/vllm/{result_file_pattern}"
        vllm_latencies = extract_values(args, vllm_file_pattern)
        client_num_list = sorted(list(vllm_latencies.keys()))

    if "fastgen" in args.backend:
        mii_file_pattern = f"{log_dir}/fastgen/{result_file_pattern}"
        mii_latencies = extract_values(args, mii_file_pattern)
        client_num_list = sorted(list(mii_latencies.keys()))

    if "aml" in args.backend:
        # TODO (lekurile): Add AML code here if/when streaming is supported
        raise NotImplementedError("Percentile latency analysis is not supported for AML.")

    for client_num in client_num_list:
        plt.figure()
        percentile = 95

        if "vllm" in args.backend:
            P50_vllm_val = np.percentile(vllm_latencies[client_num], 50)
            P90_vllm_val = np.percentile(vllm_latencies[client_num], 90)
            P95_vllm_val = np.percentile(vllm_latencies[client_num], 95)
            x1 = [1, 2.5, 4]
            y1 = [P50_vllm_val, P90_vllm_val, P95_vllm_val]
            plt.bar(x1, y1, width=0.3, label="vLLM", align="center", color="orange")

        if "fastgen" in args.backend:
            P50_mii_val = np.percentile(mii_latencies[client_num], 50)
            P90_mii_val = np.percentile(mii_latencies[client_num], 90)
            P95_mii_val = np.percentile(mii_latencies[client_num], 95)
            x2 = [1.3, 2.8, 4.3]
            y2 = [P50_mii_val, P90_mii_val, P95_mii_val]
            plt.bar(
                x2, y2, width=0.3, label="DeepSpeed-FastGen", align="center", color="blue"
            )

        if "aml" in args.backend:
            # TODO (lekurile): Add AML code here if/when streaming is supported
            raise NotImplementedError("Percentile latency analysis is not supported for AML.")

        out_file = (
            out_dir
            / f"p{percentile}_token_latency_{model}_c{client_num}_tp{tp_size}_p{prompt}g{gen}.png"
        )

        plt.ylabel("Latency (s)", fontsize=14)
        plt.legend(loc=2)

        label_x = ["P50", "P90", "P95"]
        plt.xticks([1, 2.5, 4], label_x)

        plt.title(f"Model: {model}, Clients: {client_num}, Prompt: {prompt}, Gen: {gen}, TP: {tp_size}")
        plt.savefig(out_file)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    args = get_args()

    result_params = get_result_sets(args)

    # Generate plots
    for model, tp_size, bs, replicas, prompt, gen in result_params:
        output_charts(
            args=args,
            model=model,
            tp_size=tp_size,
            bs=bs,
            replicas=replicas,
            prompt=prompt,
            gen=gen,
            log_dir=args.log_dir,
            out_dir=args.out_dir,
        )
