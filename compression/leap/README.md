# LEAP: Liberate Sparse-view 3D Modeling from Camera Poses

### [Project Page](https://hwjiang1510.github.io/LEAP/) |  [Paper](https://arxiv.org/pdf/2310.01410.pdf)
<br/>

> LEAP: Liberate Sparse-view 3D Modeling from Camera Poses

> [Hanwen Jiang](https://hwjiang1510.github.io/), [Zhenyu Jiang](https://zhenyujiang.me/), [Yue Zhao](https://zhaoyue-zephyrus.github.io/), [Qixing Huang](https://www.cs.utexas.edu/~huangqx/)


## Installation
```
conda create --name leap python=3.9
conda activate leap

# Install pytorch or use your own torch version. We use pytorch 2.0.1
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install pytorch3d, please follow https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
# We use pytorch3d-0.7.4-py39_cu117_pyt201

# (Optional) Install flash attention to enable training on limited GPU memory
# We tested with flash attention 1.0.7
# Please follow https://github.com/Dao-AILab/flash-attention
# Using flash attention during training will lead to slightly worse performance
# If you don't want to install flash attention, please comment related code in encoder.py and lifting.py

pip install -r requirements.txt 
```

## Pre-trained Weights
We provide the model weights trained on [Omniobject3D dataset](https://utexas.box.com/shared/static/8v5asrdb4wzn55atzrdy5csblasg1jdu.tar) and [Kubric ShapeNet dataset](https://utexas.box.com/shared/static/sfvznslazrwrrof8fv7uy23myc0oizhx.tar).

## Run LEAP demo
- Download pretrained weights, modify pre-trained weight path at L34 of demo.py.
- Run with `./demo.sh`.
- You can try to capture your own images, and use segmented images as inputs.


## Train LEAP

### Download Dataset
- Please follow [Omniobject3D](https://omniobject3d.github.io/), [FORGE](https://ut-austin-rpl.github.io/FORGE/) and [Zero123](https://zero123.cs.columbia.edu/) to download the three object-centric datasets.
- Please follow PixelNeRF to download DTU dataset.
- Modify `self.root` in the dataloaders.

### Training
- Use `./train.sh` and change your training config accordingly.
- The default training configurations require about 300GB at most, e.g. 8 A40 GPUs with 40GB VRAM, each.
- If you don't have enough resources, please consider using flash attention.


## Evaluate LEAP
- Use `./eval.sh` and change your evaluation config accordingly.

## Known Issues
- The model trained on Omniobject3D cannot predict densities accurately on real images, please use Kubric pre-trained weights instead.


## Citation
```bibtex
@article{jiang2022LEAP,
   title={LEAP: Liberate Sparse-view 3D Modeling from Camera Poses},
   author={Jiang, Hanwen and Jiang, Zhenyu and Zhao, Yue and Huang, Qixing},
   journal={ArXiv},
   year={2023},
   volume={2310.01410}
}
```
