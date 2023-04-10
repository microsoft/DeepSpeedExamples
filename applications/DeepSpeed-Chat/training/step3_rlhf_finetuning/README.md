# Reinforcement Learning from human feedback (RLHF) finetuning

In the reinforcement Learning from human feedback finetuning (RLHF), we take the finetuned SFT model (from step 1) and the RW (from step 2) to further finetune the SFT using the proximal policy optimization (PPO). Different than standard large language models (LLMs) finetuning, mutilple models are used here and there exists interaction between different models. Please see Figure (need to wait for Xiaoxia's figure) below. 

<img src="../image/RLHF.png" width="900"/>

There are two main challenges here: (1) how to handle the large memory consumption used for multiple models and (2) how to effeciently generate answers as it (usually) dominates the training cost in RLHF. Here, we give a brief answer of both questions.

1. **Memory management in DeepSpeed-RLHF**

We have three key techniques to reduce the memory pressure for RLHF finetuning.

First, thanks to DeepSpeed ZeRO optimization, we are able to partition both the model parameters and its optimizers across the entire training GPU system. This significantly reduced the memory consumption for those models.

Second, the reference-model has the same size as actor model in PPO training loop, which has a non-trivial memory requirement. However, the usage of it only happens when we need the old behavior probablity. As such, the compute entensive of reference model is quite low as compared to actor model. To reduce the memory pressure, we provide a single model offload option to only offload reference model to CPU. We see minimal throughput effect under the same training batch size when we offload reference model to CPU or not. However, if actor model is offloaded to CPU, the training significantly slows down.

Thrid, the optimization states of optimizer consumes a large amount of training memory. To alleviate this, we implement LoRA, which only has a small partion of parameters updated during training. Therefore, the optimization states is much smaller than standard training.

2. **DeepSpeed Hybrid Engine**

Training and inference usually utilize two different backends in most of high-optimized systems, including DeepSpeed. The reason is that these two objectives are normally used for different scenerios, i.e., training for model updating and inference for model deployment. However, this story does not hold for RLHF finetuning anymore. For each step, the actor model needs to generate the answer for the provided query. As such, standard training mode will be the bottleneck for RLHF finetuning since it is not optimized for inference.

Besides, as mentioned above, we are able to use ZeRO optimization to partition the model across different GPUs. During generation, if we have to gather the parameters across GPUs (or nodes) for each generation step, the communication cost will be very high, particularly for large models.

To overcome both challenges, we here provide DeepSpeed Hybrid Engine (DeepSpeed-HE). It can automatically switch between training engine and inference engine provided by DeepSpeed so the training can benefit both optimizations from DeepSpeed. More importanly, DeepSPeed-HE can auto-matically change ZeRO-3 training mode to Tensor Parallelism (also known as Model Parallelism) inference. As such, we remove the repeated parameter gathering and provide highly effective inference experience. As a byproduct, users are able to directly import huggingface models for training instead of modifying it for tensor parallelism or pipeline parallelism training.

## How to train RLHF
We provide multiple actor training scripts in ``training_scripts`` folder with a fixed OPT-350m reward model. Users are encourage to use different sizes reword models based on their preferrence. For instance, if you have a single GPU and want to train a OPT-1.3B model, you can simply run ``bash training_scripts/single_gpu/run_1.3b_lora.sh`` to launch the training.

## Some arguments explanation and largest model training on your own system
We provide most of unique arguments used in DeepSpeed RLHF other than the previous two steps here.
| Args                                                               | Explanation                                                                                  | Note                                                                                                                                                                     |
|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --unsupervised_dataset_name and --unsupervised_dataset_config_name | Huggingface datasets standard setting to collect the data, e.g., using Wikitext-103          | When both are provided, during each PPO training, we will also add the pretraining objective. Based on InstructGPT, this will enhance the model's benchmark performance. |
| --unsup_coef                                                       | Used to balance RLHF/PPO loss and the unsupervised loss                                      |                                                                                                                                                                          |
| --per_device_train_batch_size and --per_device_mini_batch_size     | The first one is the generation batch size and the second one is the PPO training batch size | Usually, the first one needs to be divisbale by the first one.                                                                                                           |
| --generation_batch_numbers                                         | Generated N batches then do PPO training                                                     | This setting is common in RL, i.e., we generate an experiment table then do RL training                                                                                  |
| --ppo_epochs                                                       | For the generated experiments, how many PPO epochs we want to perform                        |                                                                                                                                                                          |
| --max_prompt_seq_len and --max_answer_seq_len                      | The length of the query and the length of the answer                                         |                                                                                                                                                                          |
| --enable_hybrid_engine                                             | Enable it to use DeepSpeed Hybrid Engine                                                     | This will significantly speedup your training                                                                                                                            |
| --inference_tp_size                                                | The inference tensor parallelism size                                                        | Normally, do not exceed the size of a single node                                                                                                                        |
| --release_inference_cache                                          | Release the memory reserved for sentence generation                                          | This will slow down the training a bit but perhaps increasing the training batch size.                                                                                   |
| --unpin_actor_parameters                                           | Do not gather the actor parameter for generation                                             | This will significantly slow down the generation phase. Usually we do not recommand this option.                                                                         |
| --offload_reference_model                                          | Only offload the reference model to CPU                                                      | This helps increase the batch size with neglible time cost                                                                                                               |
| --enable_ema                                                       | Add another model to collect the expotential moving average of the actor model's weight      | According to InstructGPT, the EMA weight has better performance than actor model's final checkpoint                                                                      |

Theoretically, the largest model you can train for this step is similar to step-1 SFT finetuning if you enable (i) zero stage 3 (if you use multiple GPUs) (ii) gradient checkpoint, (iii) LoRA, (iv) reference model offloading. However, in practice, this is not the case and we are still investigating the reason. For now, we suggest users use ``Total-GPU-Memory-in-GB / 10`` billion parameter size as the upper parameters bound of the sum of actor model and critical model for safety. But users are welcome to try the real limit and also explore why we cannot train larger models.

## How to evaluate
Users are either use the ``prompt_eval.py`` from step-1 SFT to test the Q&A quality of the model or use the proof-of-concept multi-round conversation API to do the evaluation. 

## Instablity of RLHF training and others
RLHF is still a new field and as usual, we found some training instablities during our exploration. We share them here and are actively working on the solutions. Also, we look forward to solutions from the community as well. 

We find it is very unstable to use (1) different generation training batch (``--per_device_train_batch_size``) and ppo training batch size (``--per_device_mini_batch_size``), (2) larger than 1 ppo training epochs (``--ppo_epochs``), and/or (3) more than 1 genration batch sizes (``--generation_batch_numbers``). They all essentailly point to the same problem: we are not able to update the actor model for multiple times after the experiment data generation. That is to say, in all our successful trained runs, we set ``per_device_train_batch_size=per_device_mini_batch_size`` and ``ppo_epochs=generation_batch_numbers=1``. This is unexpceted for a standrad RL training pipeline, and we have tried different methods to overcome this but all failed. Among those trails, one of the most possible reason that causes this instability is that we found the ``log_probs`` and ``old_log_probs`` used in ``actor_loss_fn`` function can quickly diverge even in one iteration, which causes the corresponding ``ratio`` to be huge. By setting a strict upper bound can allevaite this problem but cannot fully resove the convergent issue. 

We also find by adding the unsupervised training is not easy to work. We tried to use the coefficient (``--unsup_coef=27.8``) provided by InstructGPT but it causes the RLHF training unstable. According to InstructGPT, the unsupervised training will mainly affect the model quality on standard benchmarks instead of the RLHF performance. We did not pay much effort to tune this parameter. 

**Others**

How to evaluate the RLHF trained model and the first step trained SFT model is still unclear. Oftentimes, researchers and practitioners rely on annotators to give the score or use the powerful trained RLHF model, e.g., ChatGPT or GPT4, to measure the quality. As there is no standard solution yet, we here do not provide a metric for our finetuned model. 

Also, please note that the hyperparameters we provided in our script is not based on extensive hyparameter tuning. Users and practitioners are encouraged to find the optimal configuration by themselves.