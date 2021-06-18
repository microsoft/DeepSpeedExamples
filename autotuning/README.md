# DeepSpeed Autotuning



## Usage

### Comannd Line Interface

Users invoke DeepSpeed autotuning by specifying `--autotuning=[run|tune]` and providing a `ds_config.json` file with `autotuning` enabled, shown as belown.

```bash
deepspeed --autotuning=[run|tune] --num_gpus=$NUM_GPUS --num_nodes=$NUM_NODES <user script> --deepspeed ds_config.json <other user args>
```

* `--autotuning=tune` returns the best deepspeed configuration

* `--autotuning=run` finds the best deepspeed configuration and launches the training with that set-up


### Autotuing Experiment Space

A default expereiment exploration space is provided in [`DEFAULT_CONFIG_SPACE`](https://github.com/microsoft/DeepSpeed-internal/blob/cheng/autotuning/deepspeed/autotuning/autotuner.py#L28).

Users can specify numbers (a value or a list of values) in the `ds_config.json` to overwrite any of the tuning parameters.

## Autotuing Configuration

```json
  "autotuning": {
    "enabled": true,
    "start_step": 5,
    "end_step": 10,
    "num_nodes": 1,
    "num_gpus": 1,
    "metric": "latency",
    "arg_mappings": {
      "train_micro_batch_size_per_gpu": "per_device_train_batch_size"
    }
  }
```
- enabled (bool): enable or disable autotuing
- start_step (int): the training step to start recording autotuing metrics
- end_step (int): the training step to end recording autotuing metrics
- num_nodes (an int or a list of ints): the number of nodes to use for a run
- num_gpus (an int or a list of ints):  the number of gpus per node to use for a run
- metric (one of ["latency" | "throughput" | "FLOPS"  | "forward" |
"backward" | "step"]): the metric to use for selecting the optimal experiment set-up
- arg_mappings (dict):

### Examples

HF DeBERTa example:

```bash
batch_size=4
TASK_NAME=mrpc
output_dir=/tmp/mrpc_out

deepspeed --autotuning tune $HF_PATH/transformers/examples/pytorch/text-classification/run_glue.py --deepspeed_config ds_config.json \
  --model_name_or_path microsoft/deberta-v2-xxlarge \
  --task_name ${TASK_NAME} \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size ${batch_size} \
  --learning_rate 13e-6 \
  --num_train_epochs 3\
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --logging_dir ${output_dir} \
  --save_steps 0
  ```

An example `ds_config.json`:

```json
{
  "train_micro_batch_size_per_gpu": [8],
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true,
    "cpu_offload": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "autotuning": {
    "enabled": true,
    "start_step": 5,
    "end_step": 10,
    "num_nodes": 1,
    "num_gpus": [2, 4],
    "metric": "latency",
    "arg_mappings": {
      "train_micro_batch_size_per_gpu": "per_device_train_batch_size"
    }
  }
}

```
## References
- https://huggingface.co/transformers/main_classes/trainer.html#deployment-with-multiple-gpus
