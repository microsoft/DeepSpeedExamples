# QANet-Pytorch
## Requirements
  * python 3.6
  * pytorch 0.4.0
  * tqdm
  * spacy 2.0.11
  * visdom

## Usage
Download the data
```bash
# download SQuAD and Glove
$ sh download.sh
```

Train the model
```bash
$ python main.py --batch_size 32 --epochs 30 --with_cuda --use_ema
```
Train the model using multi GPU
```bash
$ python main.py --batch_size 32 --epochs 30 --with_cuda --use_ema --multi_gpu
```
Debug
```bash
$ python main.py --batch_size 32 --epochs 3 --with_cuda --use_ema --debug
```
