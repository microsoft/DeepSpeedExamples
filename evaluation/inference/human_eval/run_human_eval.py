import os
import torch
import mii
import numpy
import argparse
from deepspeed.accelerator import get_accelerator
from transformers import pipeline
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, help="evaluation model name")
parser.add_argument("--max-tokens", type=int, default=256, help="max new tokens")
parser.add_argument("--num-samples-per-task", type=int, default=10, help="number of samples to gen/eval per task")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
args = parser.parse_args()

def generate_base_completion(problem_prompt: str) -> str:
    return base_pipe(problem_prompt, do_sample=True)[0]["generated_text"]

def generate_mii_completion(problem_prompt: str) -> str:
    return mii_pipe(problem_prompt, max_new_tokens=args.max_tokens)[0].generated_text

def generate_samples(generation_function):
    samples = [
        dict(task_id=task_id, completion=generation_function(problems[task_id]["prompt"])) for task_id in problems
        for _ in range(args.num_samples_per_task)
    ]
    return samples
# TODO (lekurile): Move these functions to utils in DSE repo (TBD)

print("Initializing HuggingFace Pipeline")
device = torch.device(get_accelerator().device_name(args.local_rank))
base_pipe = pipeline(model=args.model,
                        device=torch.device(get_accelerator().device_name(args.local_rank)),
                        max_length=args.max_tokens,
                        return_full_text=False)

print("Initializing DeepSpeed-MII Pipeline")
mii_pipe = mii.pipeline(args.model)

print("Loading Problems")
problems = read_problems("human-eval/data/HumanEval.jsonl.gz")

print("Generating Base Samples")
base_samples = generate_samples(generate_base_completion)

print("Generating MII Samples")
mii_samples = generate_samples(generate_mii_completion)

print("Writing Samples")
write_jsonl("base_samples.jsonl", base_samples)
write_jsonl("mii_samples.jsonl", mii_samples)

print("Evaluating Samples")
base_results = evaluate_functional_correctness("base_samples.jsonl")
mii_results = evaluate_functional_correctness("mii_samples.jsonl")

print(f"Base Results = {base_results}")
print(f"MII Results = {mii_results}")

print("Executing Assertions")
for key in base_results.keys():
    assert numpy.allclose(base_results[key], mii_results[key], rtol=0.2), \
        f"Base result: {base_results[key]}, MII result: {mii_results[key]}, outside of rtol."
