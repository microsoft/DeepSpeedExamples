# description: train Huggingface transformer using DeepSpeed
#
# In this example we train a 1.6B parameter gpt2 model using Deepspeed and
# Huggingface's transformers library.

from dataclasses import dataclass, asdict

@dataclass
class JobArguments:
    """Arguments controlling job submission to Azure ML."""

    model_checkpoint: str = "distilbert-base-uncased"
    task: str = "cola"
    node_count: int = 1
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16


def print_run_cmd(args: JobArguments):
    """Submit GLUE experiment to azureml."""

    cmd = f"""deepspeed finetune_glue.py \\
    --output_dir outputs \\
    --model_checkpoint {args.model_checkpoint} \\
    --task {args.task} \\
    --num_train_epochs {args.num_train_epochs} \\
    --per_device_train_batch_size {args.per_device_train_batch_size} \\
    --per_device_eval_batch_size {args.per_device_eval_batch_size} \\
    --disable_tqdm 1 \\
    --deepspeed ds_config.json
    """
    print("\nPlease copy and paste the command below to run the job\n")
    print(cmd)

if __name__ == "__main__":

    # https://huggingface.co/transformers/pretrained_models.html
    model_checkpoints = [
        "distilbert-base-uncased",  # 66M
        # "bert-base-uncased",  # 110M
        # "bert-large-uncased",  # 336M
        # "gpt2",  # 117M
        # "gpt2-medium",  # 345M
        # "gpt2-large",  # 774M
        # "gpt2-xl",  # 1558M
    ]

    # https://openreview.net/pdf?id=rJ4km2R5t7
    tasks = [
        # "wnli",  # 634, inference
        # "rte",  # 2.5k, inference
        # "mrpc",  # 3.7k, paraphrase
        # "stsb",  # 7k, sentence similarity
        "cola",  # 8.5k, single-sentence
        # "sst2",  # 67k, single-sentence
        # "qnli",  # 105k, inference
        # "mnli",  # 393k, inference
        # "qqp",  # 364k, paraphrase
    ]

    for model_checkpoint in model_checkpoints:
        for task in tasks:
            args = JobArguments(
                model_checkpoint=model_checkpoint,
                task=task,
            )
        
        print_run_cmd(args)
