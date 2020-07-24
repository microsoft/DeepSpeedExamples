# BERT pre-training

This repo contains DeepSpeed's version of BERT for pre-training.

Using DeepSpeed's optimized transformer kernels as the building block,
we were able to achieve the fastest BERT training record: 44 minutes on 1,024
NVIDIA V100 GPUs, compared with the previous best published result of 67 minutes on
the same number and generation of GPUs.

* Brief overview and more details, see our [press release](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/).
* Detailed technology deep dive, see our [blog post](https://www.deepspeed.ai/news/2020/05/27/fastest-bert-training.html).
* Tutorial on how to reproduce our results, see our [BERT pre-training tutorial](https://www.deepspeed.ai/tutorials/bert-pretraining/).
* The source code for our transformer kernels can be found in the [DeepSpeed repo](https://github.com/microsoft/deepspeed).


Although the fastest BERT training record was achieved using internal datasets, the DeepSpeed BERT model can also be pre-trained using datasets provided by [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT).  Instructions for preparing the datasets are available [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#dataset-guidelines). In addition, the following three files are provided in this repo to perform the full pre-training of DeepSpeed BERT with Nvidia datasets.

1. <code>ds_train_bert_nvidia_data_bsz64k_seq128.sh</code> for phase 1 training
2. <code>ds_train_bert_nvidia_data_bsz32k_seq512.sh</code> for phase 2 training
3. <code>bert_large_lamb_nvidia_data.json</code> for configuring the different parameters relating to the model, datasets, hyper-parameters, etc.

As reference, the default settings of these script and configuration files will pre-train the model to achieve EM/F1 finetuning scores of 83.57/90.62 on SQuAD.
