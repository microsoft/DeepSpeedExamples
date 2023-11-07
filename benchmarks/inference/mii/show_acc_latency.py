import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt

from postprocess_results import read_json, get_token_acc_latency

bs = 768
    
tp_sizes = {
    "7b": [1],
    "70b": [4, 8],
}

prompt_gen_pairs = [
    (1200, 60),
    (1200, 128),
    (2600, 60),
    (2600, 128),
]

PERCENTILE_VALS = [50, 90, 95, 99]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, default="logs.release")
    parser.add_argument("--out_dir", type=Path, default="charts/token_latency")
    args = parser.parse_args()
    return args


def extract_values(file_pattern, percentile):
    print(f"Looking for files matching {file_pattern}")
    files = glob.glob(file_pattern)

    print(f"Found {len(files)} files")

    latencies = {}
    for f in files:
        prof_args, response_details = read_json(f)

        acc_latency = get_token_acc_latency(response_details, percentile)

        client_num = prof_args["client_num"]
        latencies[client_num] = acc_latency

    return latencies


def output_charts(model_size, tp, bs, percentile, prompt, gen, log_dir, out_dir):
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return
    
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    mii_file_pattern = f"{log_dir}/logs.llama2-{model_size}-tp{tp}-b{bs}/llama2-{model_size}-tp{tp}-b{bs}_c*_p{prompt}_g{gen}.json"
    vllm_file_pattern = f"{log_dir}/logs.vllm-llama2-{model_size}-tp{tp}/vllm-llama2-{model_size}-tp{tp}_c*_p{prompt}_g{gen}.json"

    mii_latencies = extract_values(mii_file_pattern, percentile)
    vllm_latencies = extract_values(vllm_file_pattern, percentile)
    client_num_list = sorted(list(mii_latencies.keys()))

    for client_num in client_num_list:

        if client_num not in mii_latencies or client_num not in vllm_latencies:
            continue

        plt.figure(figsize=(6, 4))

        plt.plot(vllm_latencies[client_num][:gen + 10], label=f"vLLM")
        plt.plot(mii_latencies[client_num][:gen + 10], label=f"DeepSpeed-FastGen")

        plt.title(f'Model Llama 2 {model_size.upper()}, #Client={client_num} Prompt: {prompt}, Generation: {gen}, TP: {tp}')
        plt.xlabel('Generated token count', fontsize=14)
        plt.ylabel('Latency', fontsize=14)
        plt.legend()
        plt.grid(True)
        # plt.show()
        out_file = out_dir / f"acc_latency_llama{model_size}_pt{percentile}_c{client_num}_tp{tp}_p{prompt}g{gen}.png"
        plt.savefig(out_file)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    args = get_args()
        
    for model_size, tps in tp_sizes.items():
        for tp in tps:
            for prompt, gen in prompt_gen_pairs:
                for percentile in PERCENTILE_VALS:
                    output_charts(model_size, tp, bs, percentile, prompt, gen, args.log_dir, args.out_dir)

