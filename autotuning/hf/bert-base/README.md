# [bert-base-cased](https://huggingface.co/bert-base-cased)

This model has the following configuration:

- 12-layer
- 768 hidden dimension
- 12 attention heads
- 110M parameters.

## Environment

The training use fp32 and runs on 1 node with 16 Nvidia V100 GPUs. The autotuning uses the same hardware resource as the training. `max_train_batch_size` is not defined.

- transformers (4.12.0.dev0)
- datasets (1.11.0)

## Throughput Comparsion

The table below shows the throughput (samples per second) comparsion. The corresponding train micro batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would use in the handtuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
 - `baseline` is the vanila Hugging Face (HF) without DeepSpeed (DS) and mbs is hand-tuned.
 - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
 - `HF + DS autotuning` is HF with DS, and the DS configuration selected from autotuning.

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), graident accumulation steps (gas), train micro batch size per GPU (mbs or tmbspg).

| Model name | baseline (vanila HF)          | HF + DS handtuned                    | HF + DS autotuning           |
| ---------- | ----------------------------- | ------------------------------------ | ---------------------------- |
| BERT-base  | 2502.236 (gas = 1, mbs = 128) | 2523.684 (z = 0, gas = 1, mbs = 128) | 2682.849 (z0_gas1_tmbspg220) |

## Detailed `HF + DS autotuning` Result Summary

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed foward, backward and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

- Fast-mode Autotuning time: 43 mins
- Number of experiments: 35
- Throughput Improvement over baseline: 1.07x

| tuning_space | num_experiments | best_metric_val | best_exp_name     |
| :----------- | --------------: | --------------: | :---------------- |
| z0           |               9 |         2880.94 | z0_gas1_tmbspg220 |
| z1           |               7 |         2861.43 | z1_gas1_tmbspg220 |
| z2           |               8 |         2714.96 | z2_gas1_tmbspg240 |
| z3           |              11 |         2420.78 | z3_gas1_tmbspg240 |
| global       |              35 |         2880.94 | z0_gas1_tmbspg220 |

Tuning completed in 0:43:33.853567. Total number of experiments: 35.
