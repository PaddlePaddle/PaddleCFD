# PPDiffusion

**注意:** 本文件中的系统命令基于Linux系统。

## 代码

```sh
git clone https://github.com/PaddlePaddle/PaddleCFD.git
```

## 环境配置

```sh
pwd
cd ../../
env PYTHONPATH=$PYTHONPATH:$(pwd)   # 设置临时路径
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ./examples/ppdiffusion
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 数据

我们使用了与原文相同的数据集，该数据集来源于另一篇[论文](https://arxiv.org/abs/2108.07799)中的 Navier-Stokes 数据集。如果您想按照该论文的说明自行获取数据集, 可以跳过此部分。

### 获取我们处理过的数据

#### 如果要获取所有文件

```sh
wget https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppdiffusion/ns_trajectory_dataset.tar.gz
tar -xzvf ns_trajectory_dataset.tar.gz
```

然后, 在 `configs/config.yaml` 文件中将 `root_data_dir` 设置为 `./ns_trajectory_dataset/single/` 或 `./ns_trajectory_dataset/multi/`

#### 如果不手动下载数据集

在 `configs/config.yaml` 文件中将 `root_data_dir` 设置为 `https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppdiffusion/ns_trajectory_dataset/single/` 或 `https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppdiffusion/ns_trajectory_dataset/multi/`

### 我们的数据处理脚本

参见 `dataset_script.ipynb`。

## 运行

### 若使用我们的预训练模型

interpolation ckpt: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp.pdparams

forecast ckpt: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/forecast.pdparams

### 检查配置文件

检查 `configs/config.yaml` 文件中 `root_data_dir` 的设置。

#### 自动混合精度训练

若需减小训练显存占用，提升训练速度，可以将`configs/config.yaml` 文件中 `TRAIN.enable_amp` 设置为 true，以开启混合精度训练(AMP)。

开启后，在默认配置下，Interpolation 过程每个 epoch 训练时间缩短约 45%，显存占用减小约 15%，Forecast 过程每个 epoch 训练时间缩短约 37%，显存占用减小约 25%。

#### 分布式训练（数据并行）

若需进一步提升训练速度，可以开启分布式训练。

开启后，若使用双卡训练，在默认配置下，Interpolation 和 Forecast 过程每个 epoch 训练时间都缩短约 50%，但需要注意的是，由于需要保存中间结果在多卡间通信，以及一些其他开销，当开启分布式训练后，显存将增大。

若需开启分布式训练，需要将运行命令改为如下形式：

```python
python -m paddle.distributed.launch --gpus=0,1,2,3 python_file.py --args
```

如 Interpolation 过程的训练命令将变为：

```python
python -m paddle.distributed.launch --gpus=0,1,2,3 train.py mode=train process=interpolation
```

### Interpolation 过程

#### 训练

```python
python train.py mode=train process=interpolation
```

#### 评估

```python
python train.py mode=test process=interpolation INTERPOLATION.ckpt_no_suffix="your checkpoint path"
# 或使用预训练模型: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp"
```

### Forecast 过程

#### 训练

```python
python train.py mode=train process=dyffusion INTERPOLATION.ckpt_no_suffix="your checkpoint path"
# 或使用预训练模型: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp"
```

#### 评估

```python
python train.py mode=test process=dyffusion INTERPOLATION.ckpt_no_suffix="your checkpoint path" FORECASTING.ckpt_no_suffix="your forecast checkpoint path"
# 或使用预训练模型: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp" FORECASTING.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/forecast"
```

### 可视化

运行以下命令，结果将在 `./outputs/最新日期目录/最新时间目录/visual/` 中生成.

```python
python train.py mode=test process=dyffusion INTERPOLATION.ckpt_no_suffix="your checkpoint path" FORECASTING.ckpt_no_suffix="your forecast checkpoint path"
# 或使用预训练模型: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp" FORECASTING.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/forecast"
```

## 参考文献与引用

参考文献论文: https://arxiv.org/abs/2306.01984

参考文献代码: https://github.com/Rose-STL-Lab/dyffusion?tab=readme-ov-file

```
@inproceedings{cachay2023dyffusion,
  title={{DYffusion:} A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting},
  author={R{\"u}hling Cachay, Salva and Zhao, Bo and Joren, Hailey and Yu, Rose},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  url={https://openreview.net/forum?id=WRGldGm5Hz},
  year={2023}
}
```

数据来源论文: https://arxiv.org/abs/2108.07799

数据代码仓库: https://github.com/karlotness/nn-benchmark?tab=readme-ov-file

```
@article{nnbenchmark21,
  title={An Extensible Benchmark Suite for Learning to Simulate Physical Systems},
  author={Karl Otness and Arvi Gjoka and Joan Bruna and Daniele Panozzo and Benjamin Peherstorfer and Teseo Schneider and Denis Zorin},
  year={2021},
  url={https://arxiv.org/abs/2108.07799}
}
```
