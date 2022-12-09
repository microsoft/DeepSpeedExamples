#### Example of fine-tuning GPT using random-LTD (https://arxiv.org/abs/2211.11586)

#### Install

``pip install -r requirements.txt``

You will also need to install updated DeepSpeed version (>0.7.5), which contains the random-ltd library.

#### Key File: run_clm_no_trainer.py

The python code is modified based on huggingface (https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py). The key added feature is our random-ltd.

#### Folders (config)

* **config:** This folder provides DeepSpeed configuration, including the schedules of sequence-length and the layers applied by random-ltd.

#### bash script

* **run_base.sh/run_medium.sh**  This bash script contains jobs for training with random-ltd
* Run the job under the gpt_finetuning directory:

 ``DeepSpeedExamples/random_ltd/gpt_finetuning$ . ./bash_script/run_base.sh``


 ``DeepSpeedExamples/random_ltd/gpt_finetuning$ . ./bash_script/run_medium.sh``
 See more descriptions and results in our [tutorial page](https://www.deepspeed.ai/).
