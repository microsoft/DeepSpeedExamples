
DeepSpeed Huggingface Inference Examples

# Contents
   * [Contents](#contents)
   * [Setup](#setup)
   * [Usage](#usage)
   * [Additional Resources](#additional-resources)
      * [DeepSpeed Inference Getting Started](#ds-inference)
      * [DeepSpeed Inference Benchmarking](#ds-inference)

# Setup
Python dependencies:
<pre>
pip install -r requirements.txt
</pre>

For the `test-wav2vec.py` speech model example, you may also need to install the following generic library:
<pre>
sudo apt-get install libsndfile1-dev
</pre>

# Usage
The DeepSpeed huggingface inference inference examples are organized into the corresponding model type directories (e.g. [`'text-generation`'](./text-generation))

# Additional Resources
Information about DeepSpeed can be found at the [deepspeed.ai](https://www.deepspeed.ai) website.

##DeepSpeed Inference Getting Started
Additional information on DeepSpeed inference can be found here:
[Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)

##DeepSpeed Inference Benchmarking
DeepSpeed inference benchmarking can be found in the DeepSpeed repository:
[DeepSpeed Inference Benchmarking](https://github.com/microsoft/DeepSpeed/tree/master/benchmarks/inference)
