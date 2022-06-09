#### Install

``pip install -r requirement.txt``

You will also need to install DeepSpeed *staging_compression_library_v1* https://github.com/microsoft/DeepSpeed-internal/tree/staging_compression_library_v1

#### Key File: run_glue.py

The python code is modified based on [HuggingFace's PyTorch text_classiciation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). The key added feature is the implementation of knowledge distillation (KD)（--distill_method one_stage). If no KD, run (--distill_method zero_stage)

#### Folders (config, huggingface_models)

* **config:** this folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.
* **huggingface_models:** This folder serves the implementation of knowledge distillation. It's based on [HuggingFace's transformer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
  The change is line 383, where we output attention_scores instead of attention_prob.

#### Run Scripts (run_jobs.sh, run_jobs_internal.py)

* **run_jobs.sh**  It gives options to users who (do not) provide models. Please read the file in detail.
* **run_jobs_internal.py** It is for internal usage. We need to download models first (use this script "download_huggingface_models.sh")