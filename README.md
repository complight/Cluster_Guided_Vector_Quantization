# CGVQ: Official Implementation of Cluster-Guided Vector Quantization for 2D Gaussian-based Image Compression
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the official implementation of **CGVQ (Cluster-Guided Vector Quantization)**, developed for my Bachelor's thesis:

**From Variational Autoencoders to Codebook Quantization: Improving 2D Gaussian-based Image Compression**

Our implementation is built on top of [GaussianImage](https://github.com/Xinjie-Q/GaussianImage), and extends it with **cluster-guided, attribute-aware quantization** for improved Gaussian-based image compression.

## Quick Start

### Cloning the Repository

The repository contains submodules, thus please check it out with

```shell
# SSH
git clone <your-repo-url> --recursive
```

or

```
# HTTPS
git clone <your-repo-url> --recursive
```

After cloning the repository, you can follow these steps to train CGVQ models under different tasks.

### Requirements

```
pip install -r requirements.txt

cd gsplat
pip install .[dev] --no-build-isolation

cd ../
```

If you encounter errors while installing the packages listed in requirements.txt, you can try installing each Python package individually using pip.

Before training, you need to download the Kodak and DIV2K-validation datasets. The dataset folder is organized as follows.

```
├── dataset
│   | kodak 
│     ├── kodim01.png
│     ├── kodim02.png 
│     ├── ...
│   | DIV2K_valid_LR_bicubic
│     ├── X2
│        ├── 0801.png
│        ├── 0802.png
│        ├── ...
```

### Experiment 

Run at project root
```
# Kodak
sh ./scripts/cluster/kodak.sh /path/to/your/dataset
# DIV2K
sh ./scripts/cluster/div2k.sh /path/to/your/dataset
```

Datasets
- Kodak: https://r0k.us/graphics/kodak/
- DIV2K Validation: https://data.vision.ee.ethz.ch/cvl/DIV2K/