
# DeepSpeed Huggingface Inference Examples

# Contents
   * [Contents](#contents)
   * [Setup](#setup)
   * [Usage](#usage)
   * [Additional Resources](#additional-resources)
       * [DeepSpeed Inference](#deepspeed-inference)
       * [Benchmarking](#benchmarking)

# Setup
Python dependencies:
<pre>
pip install -r requirements.txt
</pre>

For the `test-wav2vec.py` speech model example, you may also need to install the `libsndfile1-dev` generic library:
<pre>
sudo apt-get install libsndfile1-dev
</pre>

# Usage
The DeepSpeed huggingface inference examples are organized into their corresponding model type directories (e.g. [./text-generation](./text-generation)).

Most examples can be run as follows:
<pre>deepspeed --num_gpus [number of GPUs] test-[model].py</pre>

The exception is the `test-run-generation.py` example, located in [./text-generation/run-generation-script/](./text-generation/run-generation-script). There, a shell script file exists, [test-gpt.sh](./text-generation/run-generation-script/test-gpt.sh), as an example of how to run the script.
<pre>
deepspeed --num_nodes 1 --num_gpus 1 test-run-generation.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-xl \
    --sample_input single_query.txt \
    --fp16 \
    --ds-inference
</pre>

# Additional Resources
Information about DeepSpeed can be found at the [deepspeed.ai](https://www.deepspeed.ai) website.

## DeepSpeed Inference
Additional information on DeepSpeed inference can be found here:
* [Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)

## Benchmarking
DeepSpeed inference benchmarking can be found in the DeepSpeed repository:
* [DeepSpeed Inference Benchmarking](https://github.com/microsoft/DeepSpeed/tree/master/benchmarks/inference)
