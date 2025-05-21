# PaddleCFD

## About PaddleCFD

PaddleCFD is a deep learning toolkit for surrogate modeling, equation discovery, shape optimization and flow-control strategy discovery in the field of fluid mechanics. Currently, it mainly supports surrogate modeling, including models based on fourier neural operator (FNO), diffusion model, transformer, physics-informed model and Kolmogorov-Arnold Networks (KAN).

## Code structure

- `config`: config files for different tasks
- `doc`: documentation
- `examples`: example scripts
- `ppcfd/data`: data-process source code
- `ppcfd/model`: model source code
- `ppcfd/utils`: util code

## Docker
```
wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/docker_image/dnnfluid-car_v1.0.tar
docker load -i dnnfluid-car_v1.0.tar
```
