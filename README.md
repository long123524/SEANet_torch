# SEANet

Official Pytorch Code base for "Using a semantic edge-aware multi-task neural network to delineate agricultural parcels from remote sensing images"

[Project](https://github.com/long123524/SEANet_torch.git)

## Introduction

This paper aims to develop a semantic segmentation network for agricultural parcel delineation from remote sensing images, with particular attention to extracting parcels with regularized and closed boundaries. We build a semantic edge-aware multi-task neural network, called SEANet, to enhance the extraction of local and global features regarding the edge and of thematic information regarding agricultural parcels. Compared with conventional CNNs, SEANet fully uses semantic edge features at both the local and global levels, improving the geometric accuracy of agricultural parcel delineation.

<p align="center">
  <img src="imgs/SEANet.jpg" width="800"/>
</p>


## Using the code:

The code is stable while using Python 3.7.0, CUDA >=11.0

- Clone this repository:
```bash
git clone https://github.com/long123524/SEANet_torch.git
cd SEANet_torch
```

To install all the dependencies using conda or pip:

```
PyTorch
TensorboardX
OpenCV
numpy
tqdm
```

## Preprocessing
Contour and Distance Maps are pre-computed and can be obtained from binary mask and contour. You can use matlab or python codes to obtain them.

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <train>
    ├── image
    |   ├── 001.tif
    │   ├── 002.tif
    │   ├── 003.tif
    │   ├── ...
    |
    └── mask
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── contour
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── dist_contour
    |   ├── 001.mat
    |   ├── 002.mat
    |   ├── 003.mat
    └── ├── ...
```

For test and validation datasets, the same structure as the above.

## Training and testing

1. Train the model.
```
Will be coming soon.
```
2. Evaluate.
```
Will be coming soon.
```

### Citation:
```
Li M, Long J, Stein A, et al. Using a semantic edge-aware multi-task neural network to delineate agricultural parcels from remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing.
```
