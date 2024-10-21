# Domino Example

## Install Dependency Libraries
```
pip install -r requirements.txt
```

## Prepare the Dataset
Follow the instructions from [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/tree/main/examples_deepspeed/universal_checkpointing#download-and-pre-process-training-dataset) to prepare the training dataset:
```
git clone https://github.com/microsoft/Megatron-DeepSpeed.git

wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz
python tools/preprocess_data.py \
    --input oscar-1GB.jsonl \
    --output-prefix my-gpt2 \
    --vocab-file gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 8
```

## Execute Domino Training

Adjust GPU number, dataset path, and training configuration in script if needed:

- GPUS_PER_NODE: Specifies the number of GPUs per node.
- CHECKPOINT_PATH: The path to the checkpoint, if applicable.
- VOCAB_FILE, MERGE_FILE, DATA_PATH: Paths to the dataset files.
- --micro-batch-size: Batch size per GPU node.

| Model    | Script |
| -------- | ------- |
| GPT  | pretrain_gpt.sh    |
| LLaMA | pretrain_llama.sh     |

## Advanced Usage
You can compile Pytorch and Apex from source for better performance.

### Compile PyTorch from Source
Compile PyTorch from source could enable JIT script.
```
git clone -b v2.1.0 https://github.com/pytorch/pytorch.git
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
conda install intel::mkl-static intel::mkl-include
conda install -c pytorch magma-cuda121 # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop

# Build torchvision
git clone https://github.com/pytorch/vision.git
python setup.py develop
```

## Build Apex
```
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --config-settings "--build-option=--fast_layer_norm" ./
```