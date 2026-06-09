# UPT 模型训练流程

本文档给出使用 PaddleCFD 训练 UPT 模型的完整流程，包含环境安装、ShapeNetCar 数据下载、数据预处理和启动训练。以下命令默认在 Linux + CUDA 11.8 + Python 3.10 环境下运行。

## 1. 安装环境

```bash
git clone https://github.com/PaddlePaddle/PaddleCFD.git
cd PaddleCFD
export PADDLECFD_ROOT=$PWD

conda create -n ppcfd python=3.10 -y
conda activate ppcfd

python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

wget -c https://paddle-org.bj.bcebos.com/paddlecfd/envs/open3d-0.18.0+da239b25-cp310-cp310-manylinux_2_31_x86_64.whl
python -m pip install open3d-0.18.0+da239b25-cp310-cp310-manylinux_2_31_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

wget -nc https://paddle-org.bj.bcebos.com/paddlescience/cmake-3.23.0-linux-x86_64.tar.gz
tar -zxvf cmake-3.23.0-linux-x86_64.tar.gz
rm -f cmake-3.23.0-linux-x86_64.tar.gz
export PATH=$PWD/cmake-3.23.0-linux-x86_64/bin:$PATH

cd source/ppfno_op
python -m pip install --no-build-isolation -v .
cd ../..

python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 下载并解压数据

UPT 示例使用 ShapeNetCar / MLCFD 数据集。原始数据解压后应包含 `mlcfd_data/training_data/param0` 到 `param8`。

```bash
cd "$PADDLECFD_ROOT"
mkdir -p datasets/shapenet_car
cd datasets/shapenet_car

wget -c http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip
unzip mlcfd_data.zip
rm -f mlcfd_data.zip
rm -rf __MACOSX

cd mlcfd_data/training_data
for i in 0 1 2 3 4 5 6 7 8; do
  tar -xzf param${i}.tar.gz
done
```

原始数据中有 4 个样本缺少 `quadpress_smpl.vtk`，需要删除，否则预处理后的样本数和 UPT 数据集划分不一致。

```bash
rm -rf ./param2/854bb96a96a4d1b338acbabdc1252e2f
rm -rf ./param2/85bb9748c3836e566f81b21e2305c824
rm -rf ./param5/9ec13da6190ab1a3dd141480e2c154d3
rm -rf ./param8/c5079a5b8d59220bc3fb0d224baae2a
```

## 3. 预处理数据

训练脚本默认读取仓库根目录下的 `preprocessed`。下面的命令会生成：

```text
preprocessed/param*/<sample_id>/mesh_points.th
preprocessed/param*/<sample_id>/pressure.th
preprocessed/param*/<sample_id>/sdf_res32.th
```

如果只训练 `grid32` 配置，只需要生成 32 分辨率的 SDF：

```bash
cd "$PADDLECFD_ROOT"
python examples/UPT/data/shapenetcar/preprocess.py \
  --src datasets/shapenet_car/mlcfd_data/training_data \
  --dst preprocessed \
  --resolutions 32
```

如果还要训练 `grid48` 或 `grid64` 配置，需要一起生成对应分辨率：

```bash
python examples/UPT/data/shapenetcar/preprocess.py \
  --src datasets/shapenet_car/mlcfd_data/training_data \
  --dst preprocessed \
  --resolutions 32 48 64
```

预处理完成后应输出 `processed 889 samples`。

## 4. 检查路径配置

`examples/UPT/static_config.yaml` 默认使用相对路径：

```yaml
vars:
  data: ../..
output_path: ${vars.data}/output/UPT
global_dataset_paths:
  shapenet_car: ${vars.data}
```

这表示从 `examples/UPT` 启动训练时：

- 数据读取路径为 `PaddleCFD/preprocessed`
- 日志和 checkpoint 输出到 `PaddleCFD/output/UPT`

启动训练前创建输出目录：

```bash
cd "$PADDLECFD_ROOT"
mkdir -p output/UPT
```

## 5. 启动训练

单机 4 卡训练 `grid32` UPT 配置：

```bash
cd "$PADDLECFD_ROOT/examples/UPT"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python main_train.py \
  --accelerator gpu \
  --devices 0,1,2,3 \
  --wandb_mode disabled \
  --hp yamls/shapenetcar/upt/dim768_seq1024sdf512_cnext_lr5e4_sd02_reprcnn_grn_grid32.yaml
```

单卡训练可使用：

```bash
cd "$PADDLECFD_ROOT/examples/UPT"
export CUDA_VISIBLE_DEVICES=0

python main_train.py \
  --accelerator gpu \
  --devices 0 \
  --wandb_mode disabled \
  --hp yamls/shapenetcar/upt/dim768_seq1024sdf512_cnext_lr5e4_sd02_reprcnn_grn_grid32.yaml
```

训练日志和 checkpoint 会写入 `PaddleCFD/output/UPT/stage1/<stage_id>/`。
