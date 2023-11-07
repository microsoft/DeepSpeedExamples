import glob
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from postprocess_results import read_json, get_first_token_latency


bs = 768

tp_sizes = {
    "7b": [1],
    "70b": [4, 8],
}

prompt_gen_pairs = [
    (2600, 60),
    (2600, 128),
]

# add argument to pass log directory using argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs.release")
    parser.add_argument("--out_dir", type=str, default="charts/throuput_latency")
    args = parser.parse_args()
    return args


def extract_values(file_pattern):
    print(f"Looking for files matching {file_pattern}")
    files = glob.glob(file_pattern)

    print(f"Found {len(files)} files")

    clients = []
    latencies = []
    for f in files:
        prof_args, response_details = read_json(f)
        P50_token_latency, P90_token_latency, P99_token_latency = get_first_token_latency(response_details)
        clients.append(prof_args["client_num"])
        latencies.append((P50_token_latency, P90_token_latency, P99_token_latency))

    return clients, latencies


def display_results(model_size, tp, bs, prompt, gen, log_dir, out_dir):
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory {log_dir} does not exist")
        return
    
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    mii_file_pattern = f"{log_dir}/logs.llama2-{model_size}-tp{tp}-b{bs}/llama2-{model_size}-tp{tp}-b{bs}_c*_p{prompt}_g{gen}.json"
    vllm_file_pattern = f"{log_dir}/logs.vllm-llama2-{model_size}-tp{tp}/vllm-llama2-{model_size}-tp{tp}_c*_p{prompt}_g{gen}.json"

    mii_clients, mii_latencies = extract_values(mii_file_pattern)
    vllm_clients, vllm_latencies = extract_values(vllm_file_pattern)

    sorted_indices = sorted(range(len(mii_clients)), key=lambda k: mii_clients[k])

    print(f"Prompt: {prompt}, Generation: {gen}, TP: {tp}")
    for i in sorted_indices:
        print(f"Client: {mii_clients[i]}")
        print(f"P50 latency MII: {mii_latencies[i][0]} vLLM: {vllm_latencies[i][0]}")
        print(f"P90 latency MII: {mii_latencies[i][1]} vLLM: {vllm_latencies[i][1]}")
        print(f"P99 latency MII: {mii_latencies[i][2]} vLLM: {vllm_latencies[i][2]}")


if __name__ == "__main__":

    args = get_args()
        
    for model_size, tp in tp_sizes.items():
        for prompt, gen in prompt_gen_pairs:
            display_results(model_size, tp, bs, prompt, gen, args.log_dir, args.out_dir)

