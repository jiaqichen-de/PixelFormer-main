## PixelFormerPlus and SemanticBins
This is the official PyTorch implementation of model PixelFormerPlus and SemanticBins  
The optimization for WACV 2023 paper 'Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention'.

**[Paper](https://arxiv.org/pdf/2210.09071)** <br />
## Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Testing](#testing)
6. [Pretrained Models](#pretrained models)
7. [Models](#models)


### Installation
Os ubuntu18.04
```
conda create -n pixelformer python=3.8
conda activate pixelformer
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib tqdm tensorboardX timm mmcv eiops
```
