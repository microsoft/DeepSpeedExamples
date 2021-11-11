# [gpt2-xl](https://huggingface.co/gpt2-xl)

This model has the following configuration:
- 48-layer
- 1600 hidden dimension
- 25 attention heads
- 1.5B parameters.

Refer to [GPT-2/GPT and causal language modeling](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)

## Environment

The training use fp16 and runs on 1 node with 16 Nvidia V100 GPUs. The autotuning uses the same hardware resource as the training. `max_train_batch_size` is not defined.

- transformers (4.12.0.dev0)
- datasets (1.11.0)
## Throughput Comparsion

The table below shows the throughput (samples per second) comparsion. The corresponding train micro batch size per GPU (mbs or tmbspg) and ZeRO stage used to achieve the throughput value is also shown in the parentheses. Assume the strategy users would usein the handtuning process is to start from `mbs = 1` and increase mbs by 2 each time until running out of GPU memory.
 - `baseline` is the vanila Hugging Face (HF) without DeepSpeed (DS) and mbs is hand-tuned.
 - `HF + DS hand-tuned` is HF with DS, and mbs is hand-tuned while other DS configuration uses default values.
 - `HF + DS autotuning` is HF with DS, and the DS configuration selected from autotuning.

| Model name | baseline (vanila HF) | HF + DS hand-tuned                | HF + DS autotuning (fast-mode)   |
| ---------- | -------------------- | --------------------------------- | -------------------------------- |
| GPT2-xl    | Not runnable         | Zero1 (27.462, gas = 1, mbs = 1), | Zero1 (27.497, gas = 1, mbs = 1) |

Notation: Hugging Face (HF), DeepSpeed (DS), ZeRO stage (z), graident accumulation steps (gas), train micro batch size per GPU (mbs or tmbspg).

## Detailed `HF + DS autotuning` Result Summary

Note that the performance metric used in autotuning is calculated using the timings captured within DeepSpeed foward, backward and step functions. The sum of these timings is less than the actual training step latency, thus the throughput metric values used by autotuning would be higher than the end-to-end throughput in training.

- Fast-mode Autotuning time: 21 mins
- Number of experiments: 9
- Throughput Improvement over baseline: Inf

| tuning_space | num_experiments | best_metric_val | best_exp_name   |
| :----------- | --------------: | --------------: | :-------------- |
| z1           |               3 |         40.1749 | z1_gas1_tmbspg1 |
| z2           |               3 |         33.0472 | z2_gas1_tmbspg1 |
| z3           |               3 |         12.8604 | z3_gas1_tmbspg1 |
| global       |               9 |         40.1749 | z1_gas1_tmbspg1 |

Tuning completed in 0:20:55.156000. Total number of experiments: 9.
