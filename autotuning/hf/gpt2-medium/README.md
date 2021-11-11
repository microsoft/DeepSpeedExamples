# [gpt2-medium](https://huggingface.co/gpt2-medium)

This model has the following configuration:
- 24-layer
- 1024 hidden dimension
- 16 attention heads
- 345M parameters.

Refer to [GPT-2/GPT and causal language modeling](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)

## Environment

The training use fp16 and runs on 1 node with 16 Nvidia V100 GPUs. The autotuning uses the same hardware resource as the training. `max_train_batch_size` is not defined.

- transformers (4.12.0.dev0)
- datasets (1.11.0)

## Throughput Comparsion

The table below shows the throughput (samples per second) comparsion. The corresponding train micro batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would use in the handtuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
 - `baseline` is the vanila Hugging Face (HF) without DeepSpeed (DS) and mbs is hand-tuned.
 - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
 - `HF + DS autotuning` is HF with DS, and the DS configuration selected from autotuning.

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), graident accumulation steps (gas), train micro batch size per GPU (mbs or tmbspg).

| Model name  | baseline (vanila HF)     | HF + DS hand-tuned                | HF + DS autotuning (fast-mode) |
| ----------- | ------------------------ | --------------------------------- | ------------------------------ |
| GPT2-medium | 71.61 (gas = 1, mbs = 2) | 142.211 (z = 1, gas = 1, mbs = 4) | 163.3 (z1_gas1_tmbspg6)        |

## Detailed `HF + DS autotuning` Result Summary

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed foward, backward and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

- Fast-mode Autotuning time: 25 mins
- Number of experiments: 15
- Throughput Improvement over baseline: 2.28x

| tuning_space | num_experiments | best_metric_val | best_exp_name   |
| :----------- | --------------: | --------------: | :-------------- |
| z0           |               6 |         167.688 | z0_gas1_tmbspg5 |
| z1           |               5 |          175.46 | z1_gas1_tmbspg6 |
| z2           |               3 |         161.619 | z2_gas1_tmbspg6 |
| z3           |               1 |               0 | z3_gas1_tmbspg6 |
| global       |              15 |          175.46 | z1_gas1_tmbspg6 |

Tuning completed in 0:25:18.653731. Total number of experiments: 15.
