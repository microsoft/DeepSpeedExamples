#### Example of fine-tuning GPT using random-LTD (https://arxiv.org/abs/2211.11586)

#### Install

``pip install -r requirements.txt``

You will also need to install updated DeepSpeed version (>0.7.5), which contains the random-ltd library.

#### Key File: main_cifar.py & main_imagenet.py

* main_cifar.py The python code is modified based on when do curricula work (https://github.com/google-research/understanding-curricula). 

* main_imagenet.py The python code is modified based on  https://github.com/pytorch/examples/tree/main/imagenet

The key added feature for the above two files are our deepspeed and random-ltd.
#### Folders (config)

* **config:** This folder provides DeepSpeed configuration, including the schedules of sequence-length and the layers applied by random-ltd.

#### bash script

* **run_base.sh/run_medium.sh**  This bash script contains jobs for training with random-ltd
* Run the job under the gpt2 directory:

 ``DeepSpeedExamples/random_ltd/gpt_finetuning$ . ./bash_script/run_base.sh``
 ``DeepSpeedExamples/random_ltd/gpt_finetuning$ . ./bash_script/run_medium.sh``
 See more descriptions and results in our [tutorial page](https://www.deepspeed.ai/).
