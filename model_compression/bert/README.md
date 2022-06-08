#### Install
```pip install -r requirement.txt```
you will also need to install deepspeed *staging_compression_library_v1* https://github.com/microsoft/DeepSpeed-internal/tree/staging_compression_library_v1

#### Key File: run_glue.py
The python code is modified based on  [huggingface&#39;s pytorch text_classiciation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). The key added feature is implementation of knowledage distillation (KD) （--distill_method one_stage). If no KD, run (--distill_method zero_stage)

#### Folders (config, huggingface_models)
* **config:** this folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.
* **huggingface_models**, this folder serves the implementation of knowledage distillation. It's based on [huggingface&#39;s transformer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
   The change is line 383 where we output attention_scores instead of attention_prob

#### Run Scripts (run_jobs.sh, run_jobs_internal.py)
* **run_jobs.sh**  it gives options to users who (do not) provide models. Please read the file in deatails.
* **run_jobs_internal.py** internal usage. We need to download models first (use this script "download_huggingface_models.sh")
