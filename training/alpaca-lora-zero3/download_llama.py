#!/usr/bin/evn python3
import argparse
import huggingface_hub
from pathlib import Path

REPOS = [
    "decapoda-research/llama-7b-hf",
    # "decapoda-research/llama-13b-hf",
    # "decapoda-research/llama-30b-hf",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dst_dir", type=Path)
    args = parser.parse_args()

    for repo_id in REPOS:
        dst_dir = args.dst_dir / repo_id
        huggingface_hub.snapshot_download(
            repo_id, ignore_patterns=["*.msgpack", "*.h5"],
            local_dir=dst_dir,
            resume_download=True,
        )


if __name__ == "__main__":
    main()