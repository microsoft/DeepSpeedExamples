# Training an Masked Language Model with PyTorch and Deepspeed

In this tutorial, we will create and train a Transformer encoder on the Masked Language Modeling (MLM) task. Then we will show the changes necessary to integrate Deepspeed, and show some of the advantages of doing so.

# 1. Training a Transformer Encoder (BERT / Roberta) model for MLM

## 1.0 Some Good Practices

### Version Control and Reproducibility

One of the most important parts of training ML models is for the experiments to be reproducible (either by someone else, or by you 3 months later). Some steps that help with this are:

* Use some form of version control (eg: `git`). Additionally, make sure to save the `gitdiff` and `githash`, so that anyone trying to replicate the experiment can easily do so

* Save all the hyperparameters associated with the experiment (be it taken from a config or parsed from the command line)

* Seed your random generators

* Specify all the packages and their versions. This can be a `requirements.txt` file, a conda `env.yaml` file or a `pyproject.toml` file. If you want complete reproducibility, you can also include a `Dockerfile` to specify the environemnt to run the experiments in.

In this example, the checkpoint directory we create is of the following format:

```bash
{exp_type}.{YYYY}.{MM}.{DD}.{HH}.{MM}.{SS}.{uuid}
    `-experiment-name
        |- hparams.json
        |- githash.log
        |- gitdiff.log
        `- tb_dir/
```
So that if you revisit this experiment, it is easier to understand what the experiment was about, when was it run, with what hyperparameters and what was the status of the code when the run was executed.

### Writing Unit Tests

Unit tests can help catch bugs early as well as set up a fast feedback loop. Since some experiments can take days or even weeks to run, both of these things is quite invaluable.

In this tutorial, two primary ingredients are data creating and model training. Hence, we try and test both these parts out (see [here](./tests/test_train_bert.py)):

1. Testing Data creating: Our data creating for the MLM task involves randomly masking words to generate model inputs. So, we test if the fraction of masked tokens matches what we expect from our `DataLoader` (for more details, please take a look at)

```python
def test_masking_stats(tol: float = 1e-3):
    """Test to check that the masking probabilities
    match what we expect them to be.
    """
    ...
```

2. Model training and checkpointing: Since pretraining experiments are usually quite expensive (take days / weeks to complete), chances are that you might run into some hardware failure before the training completes. Thus, it is crucial that the checkpointing logic is correct to allow a model to resume training. One way to do so is to train a small model for few iterations and see if the model can resume training and if the checkpoints are loaded correctly. See `test_model_checkpointing` for an example test.

---

💡 **_Tip:_** Make sure to also save the optimizer state_dict along with the model parameters !

---

## 1.1 The Masked Language Modeling Task

The main idea behind the MLM task is to get the model to fill in the blanks based on contextual clues present **both before and after** the blank. Consider, for example, the following sentence:

> In the beautiful season of ___ the ___ shed their leaves. 

Given the left context of `season` and the right context of `shed their leaves`, one can guess that the blanks are `Autumn` and `trees` respectively. This is exactly what we want the model to do: utilize both left and right context to be able to fill in blanks.

In order to do that, we carry out the following steps

1. Tokenize a sentence into word(pieces)
2. Randomly select some words to mask, and replace them with a special \<Mask\> token
    * Of the masked tokens, it is common to replace a fraction of them with a random token, and leave a fraction of them unchanged.
3. Collect the actual words that were masked, and use that as targets for the model to predict against:
    * From the model's perspective, this is a simple `CrossEntropy` loss over the vocabulary of the model.

In this tutorial, we use the [`wikitext-2-v1`](https://huggingface.co/datasets/wikitext) dataset from [HuggingFace datasets](https://github.com/huggingface/datasets). To see how this is done in code, take a look at `masking_function` in [train_bert.py](./train_bert.py).


## 1.2 Creating a Transformer model

A Transformer model repeatedly applies a (Multi-Headed) Self-Attention block and a FeedForward layer to generate contextual representations for each token. Thus, the key hyperparameters for a Transformer model usually are

1. The number of Self-Attention + FeedForward blocks (depth)
2. The size of the hidden representation
3. The number of Self Attention Heads
4. The size of the intermediate representation between in the FeedForward block

Check out the `create_model` function in [train_bert.py](./train_bert.py) to see how this is done in code.

---
📌 **Note:** You can check out [[1](#1), [2](#2)] as a starting point for better understanding Transformers. Additionally, there are a number of blogs that do nice deep dive into the workings of these models (eg: [this](https://nlp.seas.harvard.edu/2018/04/03/attention.html), [this](https://jalammar.github.io/illustrated-bert/) and [this](https://jalammar.github.io/illustrated-transformer/)).

---

## 1.3 Training the Model

In order to train the model, you can run the following 

```bash
python train_bert.py --checkpoint_dir ./experiments
```
This will create a model with the default parameters (as specified by the arguments to the `train` function), and train it on the wikitext dataset. Other parameters can be configured from the command line as:

```bash
python train_bert.py --checkpoint_dir ./experiments \
    --mask_prob ${mask_prob} \
    --random_replace_prob ${random_replace_prob} \
    --unmask_replace_prob ${unmask_replace_prob} \
    --max_seq_length ${max_seq_length} \
    --tokenizer ${tokenizer} \
    --num_layers ${num_layers} \
    --num_heads ${num_heads} \
    --ff_dim ${ff_dim} \
    --h_dim ${h_dim} \
    --dropout ${dropout} \
    --batch_size ${batch_size} \
    --num_iterations ${num_iterations} \
    --checkpoint_every ${checkpoint_every} \
    --log_every ${log_every} \
    --local_rank ${local_rank}

```

The parameters are explained in more details in the doctstring of `train`.

---
💡 **_Tip:_** If you have a GPU available, you can use it to considerably speedup your training. Simply set the `local_rank` to the GPU you want to run it on. Eg: for a single GPU machine this would look like
```bash
--local_rank 0
```

---

## 2. Integrating Deepspeed For More Efficient Training


## References
> <a id="1">[1]</a> 
[Vaswani et. al. Attention is all you need. 
In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17)](https://arxiv.org/pdf/1706.03762.pdf)

> <a id="2">[2]</a>
[Devlin, Jacob et. al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT'19)](https://aclanthology.org/N19-1423.pdf)

---------

## Scratch pad (TODO Remove)
good practices
  * experiment directory saving training metadata
  * pytests

data
  * what is masked LM
  * what does the code do (link to code)

model
  * core params for transformer model (e.g., #layers, attn)

how to run
  * launching on CPU (slow) launch on single GPU (fast)
  * different train params

------

deepspeed additions
  * deepspeed.init, training loop, ckpt changes

launching across multiple GPUs

fp16
  * how to enable
  * show memory reduction when enabled via nvidia-smi
  * brief overview of how fp16 training works (e.g., loss scaling)

zero
  * introduce how zero reduces memory
  * introduce zero offload
  * update config to use z1 + offload to showcase a model that can only run with offload enabled
