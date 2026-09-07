# UPT 模型训练流程

## 模型简介

UPT（Universal Physics Transformer）是一类面向物理场预测的 Transformer 代理模型，适合处理 CFD 等仿真任务中常见的非结构网格、点云采样和规则网格特征。它的目标是在给定几何、网格点位置或历史物理状态后，直接预测查询位置上的物理量，从而替代部分高成本数值仿真流程。

本示例主要以 ShapeNetCar 气动压力预测为例：模型读取车体表面网格点、SDF 规则网格特征和查询点位置，输出每个查询点的压力值。因此，UPT 的核心特点是把不同来源的物理/几何信息统一表示为 token，再通过 Transformer 建模全局依赖，并支持在任意查询点上输出物理场结果。

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

### 可选编译器模式

`--compiler none` 使用默认动态图训练：

```bash
python main_train.py \
  --accelerator gpu \
  --devices 0 \
  --wandb_mode disabled \
  --compiler none \
  --hp yamls/shapenetcar/upt/dim768_seq1024sdf512_cnext_lr5e4_sd02_reprcnn_grn_grid32.yaml
```

`--compiler cinn` 使用 `paddle.jit.to_static(..., backend="CINN")`：

```bash
export FLAGS_enable_pir_api=true
export FLAGS_prim_enable_dynamic=true
export FLAGS_prim_all=true
export FLAGS_use_cinn=true
export ENABLE_FALL_BACK=1

python main_train.py \
  --accelerator gpu \
  --devices 0 \
  --wandb_mode disabled \
  --compiler cinn \
  --hp yamls/shapenetcar/upt/dim768_seq1024sdf512_cnext_lr5e4_sd02_reprcnn_grn_grid32.yaml
```

不传 `--compiler` 时保持原有配置行为。`ENABLE_FALL_BACK=1` 允许暂不支持的算子回退到 Paddle 执行，支持的子图仍由 CINN 编译。

### 动态图与 CINN 性能对比

下面的脚本会在同一张 GPU 上依次运行动态图和 CINN，并比较排除首个预热 epoch 后的稳定 update 时间与吞吐量：

```bash
python benchmark_cinn.py \
  --device 0 \
  --warmup_epochs 1 \
  --hp yamls/shapenetcar/upt/dim768_seq1024sdf512_cnext_lr5e4_sd02_reprcnn_grn_grid32.yaml
```

快速验证两种模式都能运行时，可以添加 `--mindurationrun`。正式性能对比应使用完整训练时长，并确保测试期间 GPU 没有其他任务。结果会保存到：

```text
PaddleCFD/output/UPT/benchmarks/<timestamp>/result.json
```

训练日志和 checkpoint 会写入 `PaddleCFD/output/UPT/stage1/<stage_id>/`。

## 6. 评估与推理

训练完成后，先确认输出目录中存在 `hp_resolved.yaml` 和 checkpoint 文件：

```text
PaddleCFD/output/UPT/stage1/<stage_id>/hp_resolved.yaml
PaddleCFD/output/UPT/stage1/<stage_id>/checkpoints/* cp=best_model.loss.test.total model.th
```

其中 `<stage_id>` 是训练日志中打印的 `stage_id`，例如：

```text
PaddleCFD/output/UPT/stage1/onmvsf6n
```

评估脚本为 `examples/UPT/eval.py`，默认支持 ShapeNetCar 的 `rans_simformer_nognn_sdf_model`。下面命令默认使用训练过程中保存的最佳 test loss checkpoint：

```text
best_model.loss.test.total
```
已有模型checkpoint样例在 https://aistudio.baidu.com/modelsdetail/45954/space

### 单样本推理

```bash
cd "$PADDLECFD_ROOT/examples/UPT"
export CUDA_VISIBLE_DEVICES=0

python eval.py \
  --run_dir ../../output/UPT/stage1/<stage_id> \
  --checkpoint best_model.loss.test.total \
  --split test \
  --sample_idx 0 \
  --device gpu:0
```

单样本结果默认保存到：

```text
PaddleCFD/output/UPT/stage1/<stage_id>/inference/test_000000.npz
```

`.npz` 文件包含 `query_pos`、`prediction`、`target`、`abs_error` 以及该样本的 MSE、RMSE、MAE、relative L2 等指标。

### 小批量评估

建议先评估少量样本，确认 checkpoint 和数据路径无误：

```bash
cd "$PADDLECFD_ROOT/examples/UPT"
export CUDA_VISIBLE_DEVICES=0

python eval.py \
  --run_dir ../../output/UPT/stage1/<stage_id> \
  --checkpoint best_model.loss.test.total \
  --split test \
  --device gpu:0 \
  --eval \
  --start_idx 0 \
  --num_samples 10 \
  --print_every 1
```

### 完整 test 集评估

```bash
cd "$PADDLECFD_ROOT/examples/UPT"
export CUDA_VISIBLE_DEVICES=0

python eval.py \
  --run_dir ../../output/UPT/stage1/<stage_id> \
  --checkpoint best_model.loss.test.total \
  --split test \
  --device gpu:0 \
  --eval \
  --print_every 10
```

完整评估默认输出到：

```text
PaddleCFD/output/UPT/stage1/<stage_id>/inference/test_best_model.loss.test.total/
```

其中：

- `metrics_test_best_model.loss.test.total.json` 保存整体指标，包括 MSE、RMSE、MAE、global relative L2、mean relative L2 等。
- `metrics_test_best_model.loss.test.total_per_sample.csv` 保存逐样本指标，便于排序查看最好 / 最差样本。

如果需要同时保存每个样本的预测结果，添加 `--save_predictions`：

```bash
python eval.py \
  --run_dir ../../output/UPT/stage1/<stage_id> \
  --checkpoint best_model.loss.test.total \
  --split test \
  --device gpu:0 \
  --eval \
  --print_every 10 \
  --save_predictions
```

开启 `--save_predictions` 后，会在输出目录下额外生成 `predictions/*.npz`，完整 test 集会占用更多磁盘空间。
