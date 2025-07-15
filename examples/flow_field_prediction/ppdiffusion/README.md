# PPDiffusion

**Attention:** The system commands in the file are based on Linux.

## Code

```sh
git clone https://github.com/PaddlePaddle/PaddleCFD.git
```

## Envs

```sh
pwd
cd ../../
env PYTHONPATH=$PYTHONPATH:$(pwd)   # set temporary path
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ./examples/ppdiffusion
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Introduction

This case is used to solve the problem of flow field prediction. The case is based on the flow around cylinders. For a detailed introduction, please refer to the original paper [paper](https://arxiv.org/abs/2306.01984).

## Data

We use the same data as the original paper, while is a Navier-Stokes dataset form another [Paper](https://arxiv.org/abs/2108.07799). If you want to get the dataset by following the instructions given by the paper and skip this part.

### Get the data we processed

#### If you want to get all files

```sh
wget https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppdiffusion/ns_trajectory_dataset.tar.gz
tar -xzvf ns_trajectory_dataset.tar.gz
```

then, set `root_data_dir` in configs/config.yaml to `./ns_trajectory_dataset/single/` or `./ns_trajectory_dataset/multi/`

#### If you don't want to download the dataset manually

set `root_data_dir` in configs/config.yaml to `https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppdiffusion/ns_trajectory_dataset/single/` or `https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppdiffusion/ns_trajectory_dataset/multi/`

### Our processing scripts

See `dataset_script.ipynb`. Attention, the dataset provided above has already been processed and there is no need to run the code in this file.

## Run

### If you want to use our pretrained model

interpolation ckpt: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp.pdparams

forecast ckpt: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/forecast.pdparams

### Check config at first

Check setting of `root_data_dir` in configs/config.yaml.

### Automatic Mixed Precision

If you need to reduce the training memory usage and increase the training speed, you can set `TRAIN.enable_amp` to true in the `configs/config.yaml` file to enable Automatic Mixed Precision (AMP).

After enabling it, under the default configuration, the training time of each epoch of the Interpolation process is shortened by about 45%, the memory usage is reduced by about 15%, and the training time of each epoch of the Forecast process is shortened by about 37%, and the memory usage is reduced by about 25%.

### Distributed training (data parallelism)

To further accelerate training speed, you can enable distributed training.

When enabled (e.g., using dual GPUs), under default configurations, the training time per epoch for both Interpolation and Forecast processes is reduced by approximately 50%. However, note that due to the communication overhead of saving intermediate results across multiple GPUs and other additional costs, enabling distributed training will increase GPU memory consumption.

To activate distributed training, modify the running command as follows:

```python
python -m paddle.distributed.launch --gpus=0,1,2,3 python_file.py --args
```

For example, the training command for the Interpolation process under four parallel cards becomes:

```python
python -m paddle.distributed.launch --gpus=0,1,2,3 train.py mode=train process=interpolation
```

### Interpolation process

#### Train

```python
python train.py mode=train process=interpolation
```

#### Eval

```python
python train.py mode=test process=interpolation INTERPOLATION.checkpoint="your checkpoint path"
# or using our pretrained checkpoint, download it and set the parameters to INTERPOLATION.checkpoint="/path/interp.pdparams"
```

### Forecast process

#### Train

```python
python train.py mode=train process=dyffusion INTERPOLATION.checkpoint="your checkpoint path"
# or using our pretrained checkpoint, download it and set the parameters to INTERPOLATION.checkpoint="/path/interp.pdparams"
```

#### Eval

```python
python train.py mode=test process=dyffusion INTERPOLATION.checkpoint="your checkpoint path" FORECASTING.checkpoint="your forecast checkpoint path"
# or using our pretrained checkpoint, download it and set the parameters to INTERPOLATION.checkpoint="/path/interp.pdparams" FORECASTING.checkpoint="/path/forecast.pdparams"
```

### Visulization

Run the following command and results will be found in `./outputs/the lasted date directory/the lasted time directory/visual/`.

```python
python train.py mode=test process=dyffusion INTERPOLATION.checkpoint="your checkpoint path" FORECASTING.checkpoint="your forecast checkpoint path"
# or using our pretrained checkpoint, download it and set the parameters to INTERPOLATION.checkpoint="/path/interp.pdparams" FORECASTING.checkpoint="/path/forecast.pdparams"
```

# References and citations

Reference paper: https://arxiv.org/abs/2306.01984

Reference code: https://github.com/Rose-STL-Lab/dyffusion?tab=readme-ov-file

```
@inproceedings{cachay2023dyffusion,
  title={{DYffusion:} A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting},
  author={R{\"u}hling Cachay, Salva and Zhao, Bo and Joren, Hailey and Yu, Rose},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  url={https://openreview.net/forum?id=WRGldGm5Hz},
  year={2023}
}
```

Data source paper: https://arxiv.org/abs/2108.07799

Data code repository :https://github.com/karlotness/nn-benchmark?tab=readme-ov-file

```
@article{nnbenchmark21,
  title={An Extensible Benchmark Suite for Learning to Simulate Physical Systems},
  author={Karl Otness and Arvi Gjoka and Joan Bruna and Daniele Panozzo and Benjamin Peherstorfer and Teseo Schneider and Denis Zorin},
  year={2021},
  url={https://arxiv.org/abs/2108.07799}
}
```
