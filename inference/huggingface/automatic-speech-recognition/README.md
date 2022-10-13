
# DeepSpeed Huggingface Automatic Speech Recognition Examples

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
Examples can be run as follows:
<pre>deepspeed --num_gpus [number of GPUs] test-[model].py</pre>
