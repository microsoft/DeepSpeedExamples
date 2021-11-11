# [bert-large-uncased](https://huggingface.co/bert-large-uncased)

This model has the following configuration:

- 24-layer
- 1024 hidden dimension
- 16 attention heads
- 336M parameters

The training use fp32 and runs on 1 node with 16 Nvidia V100 GPUs. The autotuning uses the same hardware resource as the training. `max_train_batch_size` is not defined.

- transformers (4.12.0.dev0)
- datasets (1.11.0)

## Throughput Comparsion

The table below shows the throughput (samples per second) comparsion. The corresponding train micro batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would usein the handtuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
 - `baseline` is the vanila Hugging Face (HF) without DeepSpeed (DS) and mbs is hand-tuned.
 - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
 - `HF + DS autotuning` is HF with DS, and the DS configuration selected from autotuning.

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), graident accumulation steps (gas), train micro batch size per GPU (mbs or tmbspg).

| Model name | baseline (vanila HF)        | HF + DS handtuned                 | HF + DS autotuning         |
| ---------- | --------------------------- | --------------------------------- | -------------------------- |
| BERT-large | 742.692 (gas = 1, mbs = 64) | 766.929 (z = 1, gas =1, mbs = 64) | 808.168 (z1_gas1_tmbspg93) |

## Detailed `HF + DS autotuning` Result Summary

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed foward, backward and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

- Fast-mode Autotuning time: 36 mins
- Number of experiments: 22
- Throughput Improvement over baseline: 1.09x

| tuning_space | num_experiments | best_metric_val | best_exp_name    |
| :----------- | --------------: | --------------: | :--------------- |
| z0           |               6 |         835.244 | z0_gas1_tmbspg93 |
| z1           |               6 |         842.243 | z1_gas1_tmbspg93 |
| z2           |               9 |         764.524 | z2_gas1_tmbspg94 |
| z3           |               1 |               0 | z3_gas1_tmbspg94 |
| global       |              22 |         842.243 | z1_gas1_tmbspg93 |

Tuning completed in 0:36:16.261417. Total number of experiments: 23.
