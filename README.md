# PaddleCFD

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-dfd.svg"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg"></a>
    <!-- <a href=""><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleCFD?color=ccf"></a> -->
    <a href="PyPI Downloads"><img src=https://static.pepy.tech/personalized-badge/ppcfd?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads></a>
</p>


## About PaddleCFD

PaddleCFD is a deep learning toolkit for surrogate modeling, equation discovery, shape optimization and flow-control strategy discovery in the field of fluid mechanics. Currently, it mainly supports surrogate modeling, including models based on Fourier Neural Operator (FNO), Transformer, Diffusion Model (DM),  Kolmogorov-Arnold Networks (KAN) and DeepONet.

<img src="./doc/paddlecfd_architecture.jpg" alt="This is an image" title="PaddleCFD architecture">


## Code structure

- `doc`: documentation
- `examples`: example scripts
- `ppcfd/data`: data-process source code
- `ppcfd/model`: model source code
- `ppcfd/utils`: utils code
- `source`: source code of paddlepaddle custom operators

## How to run on NVIDIA GPU

### Installation

##### Image pulling & container running

```bash 
# Pull docker image
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6

# Run docker container
nvidia-docker run --name ppcfd-container -v /home/:/home --network=host -it  --shm-size 64g ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash
```

##### Conda environment installation

```bash
# Clone PaddleCFD
git clone https://github.com/PaddlePaddle/PaddleCFD.git
cd PaddleCFD

# Create conda environment
conda create --name ppcfd python=3.10
conda activate ppcfd

python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Download and install paddle-backended Open3D
wget https://paddle-org.bj.bcebos.com/paddlecfd/envs/open3d-0.18.0+da239b25-cp310-cp310-manylinux_2_31_x86_64.whl
python -m pip install open3d-0.18.0+da239b25-cp310-cp310-manylinux_2_31_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# Compile customed operator to conda environment
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/cmake-3.23.0-linux-x86_64.tar.gz
tar -zxvf cmake-3.23.0-linux-x86_64.tar.gz
rm -f cmake-3.23.0-linux-x86_64.tar.gz
PATH=$PWD/cmake-3.23.0-linux-x86_64/bin:$PATH
cd source/ppfno_op
python -m pip install --no-build-isolation -v .
```

##### PaddleCFD package installation (Choose one of the following)
```bash
# Install PaddleCFD from sourcecode at PaddleCFD root directory
python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install PaddleCFD from pypi
python -m pip install ppcfd -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Quick start
```bash
# Run examples
cd PaddleCFD/examples/xxx/xxx
run the example according to the example README.md
```
## How to run on MetaX

### Quick start

Following the [guidelines of MetaX](./doc/metax/README.md) to run PaddleCFD on MetaX machine.

### Parallel efficiency on MetaX

Parallel efficiency ($\eta$) calculation,

$\eta=\frac{t_1/t_n}{n} \times 100\mathrm{\%}$

where $t_1$ is the running time on one card, $t_n$ is the running time on $n$ cards, and $n$ is the number of cards working parallelly.

| **Model** | **Single card/s** | **Single 8-card node/s** | **Four 8-card nodes/s** | **$\eta$ on single 8-card node/%** | **$\eta$ on four 8-card nodes/%** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PPFNO | 599.51 | 75.57 | 19.01 | 99.16 | 98.56 |

### [About MetaX](https://www.metax-tech.com/en/about/about.html)

## APIs

[ppcfd/data](./doc/data.md)

## Community
Join PaddleCFD WeChat group to discuss with us!

<img src="./doc/飞桨AI4S%20&%20PaddleCFD技术交流群.jpg" alt="This is an image" title="PaddleCFD Weichat" style="width: 30%;">

## License

PaddleCFD is provided under the [Apache-2.0 license](./LICENSE)
