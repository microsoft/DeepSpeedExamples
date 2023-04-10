# 🚩Benchmark setting used in [blog](https://www.deepspeed.ai/2023/04/10/deepspeed-chat.html) and [Landing Page](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/README.md)

📍An apples-to-apples comparison is critical for the machine learning community, particularly for benchmarking. Therefore, we provide detailed instructions on how we chose our settings and encourage others to compare DeepSpeed-RLHF under the same settings or adjust it based on their own settings, instead of simply comparing our results under different settings.

📍We randomly select 40% training data from the six training datasets, i.e.,

```text
Dahoas/rm-static
Dahoas/full-hh-rlhf
Dahoas/synthetic-instruct-gptj-pairwise
yitingxie/rlhf-reward-datasets 
openai/webgpt_comparisons stanfordnlp/SHP
```

The total training samples we have here is 264292. We fix the query (prompt) sequence length as **256** and generate fixed-length answer with **256** tokens. As such, the total training tokens per epoch is 135,317,504. During benchmark testing, we set the training epoch number as 1.

📍As mentioned in the instability of RLHF training tutorial, ([Tutorial](./README.md#🙋-instablity-of-rlhf-training-and-others)), we found that it is not stable to update the actor model multiple times using the generated data. Therefore, we set per_device_train_batch_size equal to per_device_mini_batch_size and ppo_epochs equal to generation_batch_numbers equal to 1 for all of our benchmark results. During testing, we also set an upper bound for the maximum global training tokens at 524,288 (batch size of 1024 with a sequence length of 512). This is the largest batch size we found during our exploration that provides a stable RLHF training experience. Users and practitioners may find better training hyperparameters to further increase this. Additionally, during testing, whenever the global training token batch size does not exceed our limit of 524,288, we always use the largest training batch size that results in an out-of-memory error to benchmark the time.

📍We hope this clearly explains our benchmark settings, and please do not hesitate to contact us if you need more information
