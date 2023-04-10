# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
Run all steps with default settings:
$ python3 run_example.py

Change the model used for each step:
$ python3 run_example.py --step-1-model 350m --step-2-model 1.3b --step-3-model 125m

Change the ZeRO stage used for step 1 & 2:
$ python3 run_example.py --step-1-zero-stage 1 --step-2-zero-stage 3

Run a subset of the steps:
$ python3 run_example.py --steps-to-run 1 2

Note: Step 3 relies on models trained in Steps 1 & 2. If you have already
trained these models, you can run just Step 3 and select which models from
Steps 1 & 2 to use. For example, let's train models for Steps 1 & 2 using
125m and 350m models:
$ python3 run_example.py --steps-to-run 1 2 --step-1-model 125m --step-2-model 125m
$ python3 run_example.py --steps-to-run 1 2 --step-1-model 350m --step-2-model 350m

Now we can run Step 3 with any combination of these models:
$ python3 run_example.py --steps-to-run 3 --step-1-model 125m --step-2-model 350m
$ python3 run_example.py --steps-to-run 3 --step-1-model 350m --step-2-model 125m
"""

import argparse
import subprocess
import os
import datetime
import time

step_dirs = {
    1: "step1_supervised_finetuning",
    2: "step2_reward_model_finetuning",
    3: "step3_rlhf_finetuning",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps-to-run",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=(1, 2, 3),
        help="Which steps of the ChatGPT example to run",
    )
    parser.add_argument(
        "--step-1-model", type=str, default="125m", help="Which model to use for step 1"
    )
    parser.add_argument(
        "--step-2-model", type=str, default="125m", help="Which model to use for step 2"
    )
    parser.add_argument(
        "--step-3-model", type=str, default="125m", help="Which model to use for step 3"
    )
    parser.add_argument(
        "--step-1-zero-stage",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="ZeRO stage for step 1 (Actor) training",
    )
    parser.add_argument(
        "--step-2-zero-stage",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="ZeRO stage for step 2 (Critic) training",
    )
    parser.add_argument(
        "--output-dir",
        type=lambda x: os.path.abspath(x),
        default="./example_output",
        help="Directory for output of each step",
    )
    args = parser.parse_args()
    return args


def get_model_size(args, step_num):
    return getattr(args, f"step_{step_num}_model")


def get_zero_stage(args, step_num):
    return getattr(args, f"step_{step_num}_zero_stage")


def get_output_dir(args, step_num):
    model_size = get_model_size(args, step_num)
    output_dir = os.path.join(args.output_dir, f"step_{step_num}", f"{model_size}")
    if step_num in (1, 2):
        zero_stage = get_zero_stage(args, step_num)
        output_dir = os.path.join(output_dir, f"stage{zero_stage}")
    return output_dir


def get_script(args, step_num):
    model_size = get_model_size(args, step_num)
    return os.path.join(os.getcwd(), step_dirs[step_num], f"run{model_size}.sh")


def verify_model(args, step_num):
    output_dir = get_output_dir(args, step_num)
    zero_stage = get_zero_stage(args, step_num)
    model_size = get_model_size(args, step_num)
    model_file = os.path.join(output_dir, "pytorch_model.bin")
    if not os.path.isfile(model_file):
        error_str = f"Step {step_num} model has not been trained. Train it with:\n"
        error_str += f"python3 run_example.py --steps-to-run {step_num}"
        error_str += f" --step-{step_num}-model {model_size}"
        error_str += f" --step-{step_num}-zero-stage {zero_stage}"
        raise RuntimeError(error_str)


def get_cmd(args, step_num):
    output_dir = get_output_dir(args, step_num)
    script = get_script(args, step_num)

    if step_num in (1, 2):
        zero_stage = getattr(args, f"step_{step_num}_zero_stage")
        cmd = f"bash {script} {output_dir} {zero_stage}"
    if step_num == 3:
        verify_model(args, 1)  # Verify step 1 model exists
        verify_model(args, 2)  # Verify step 2 model exists
        s1_dir, s1_zs = get_output_dir(args, 1), get_zero_stage(args, 1)
        s2_dir, s2_zs = get_output_dir(args, 2), get_zero_stage(args, 2)
        cmd = f"bash {script} {output_dir} {s1_dir} {s1_zs} {s2_dir} {s2_zs}"

    return cmd


def main(args):
    start_time = time.time()
    for step_num in args.steps_to_run:
        print(f"---=== Running Step {step_num} ===---")
        step_start_time = time.time()

        working_dir = step_dirs[step_num]
        cmd = get_cmd(args, step_num)
        p = subprocess.Popen(cmd, cwd=working_dir, shell=True)
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(
                f"Step {step_num} exited with non-zero status {p.returncode}"
            )
        step_time = int(time.time() - start_time)
        time_str = str(datetime.timedelta(seconds=step_time))
        print(f"---=== Finished Step {step_num} in {time_str} ===---")
    total_time = int(time.time() - start_time)
    time_str = str(datetime.timedelta(seconds=total_time))
    if len(args.steps_to_run) > 1:
        print(f"---=== Finished Steps {args.steps_to_run} in {time_str} ===---")


if __name__ == "__main__":
    args = parse_args()
    main(args)
