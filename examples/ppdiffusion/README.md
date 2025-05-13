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

See `dataset_script.ipynb`.

## Run

### If you want to use our pretrained model

interpolation ckpt: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp.pdparams

forecast ckpt: https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/forecast.pdparams

### Check config at first

Check setting of `root_data_dir` in configs/config.yaml.

### Interpolation process

#### Train

```sh
env PYTHONPATH=$PYTHONPATH:$(pwd)
python train.py mode=train process=interpolation
```

#### Eval

```python
python train.py mode=eval process=interpolation INTERPOLATION.ckpt_no_suffix="your checkpoint path"
# or using our pretrained checkpoint: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp.pdparams"
```

### Forecast process

#### Train

```python
python train.py mode=train process=dyffusion INTERPOLATION.ckpt_no_suffix="your checkpoint path"
# or using our pretrained checkpoint: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp.pdparams"
```

#### Eval

```python
python train.py mode=eval process=dyffusion INTERPOLATION.ckpt_no_suffix="your checkpoint path" FORECASTING.ckpt_no_suffix="your forecast checkpoint path"
# or using our pretrained checkpoint: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp.pdparams" FORECASTING.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/forecast.pdparams"
```

### Visulization

Run the following command and results will be found in `./outputs/the lasted date directory/the lasted time directory/visual/`.

```python
python train.py mode=test process=dyffusion INTERPOLATION.ckpt_no_suffix="your checkpoint path" FORECASTING.ckpt_no_suffix="your forecast checkpoint path"
# or using our pretrained checkpoint: INTERPOLATION.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/interp.pdparams" FORECASTING.ckpt_no_suffix="https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/ppdiffusion/forecast.pdparams"
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
