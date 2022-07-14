#### Install

``pip install -r requirement.txt``

You will also need to install updated DeepSpeed verion, which contains the compression libarary


#### Key File: run_clm_no_trainer.py

The python code is modified based on huggingface (https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py). The key added feature is the compression pipeline

#### Folders (config)

* **config:** This folder provides DeepSpeed configuration, including quantization, pruning and layer reduction.

#### bash script 
* **run_zero_quant.sh**  This bash script contains jobs for training a checkpoint and then compressing this checkpoint.  Run the job under the gpt2 directory:

```DeepSpeedExamples/model_compression/gpt2$ . ./bash_script/run_zero_quant.sh```
