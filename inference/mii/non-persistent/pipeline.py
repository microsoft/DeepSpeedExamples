import argparse
from mii import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--prompts", type=str, nargs="+", default=["DeepSpeed is"])
args = parser.parse_args()

pipe = pipeline(parser.model)
responses = pipe(args.prompts, max_new_tokens=128, return_full_text=True)

if pipe.is_rank_0:
    for r in responses:
        print(r, "\n", "-" * 80, "\n")
