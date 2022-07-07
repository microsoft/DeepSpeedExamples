#### Install

``pip install -r requirement.txt``

You will also need to install DeepSpeed *staging_compression_library_v1* https://github.com/microsoft/DeepSpeed-internal/tree/staging_compression_library_v1

#### Key File: run_glue_no_trainer.py

The python code is modified based on [HuggingFace&#39;s PyTorch text_classiciation](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). The key added feature is the implementation of knowledge distillation (KD)（--distill_method one_stage). If no KD, run (--distill_method zero_stage)

#### Folders (config, huggingface_transformer, bash_script)

* **config:** This folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.
* **huggingface_transformer:** This folder serves the implementation of knowledge distillation. It's based on [HuggingFace&#39;s transformer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
  The change is line 383, where we output attention_scores instead of attention_prob.
* **bash_script**  This folder contains many bash scripts for various kinds of compression. 
