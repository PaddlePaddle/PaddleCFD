# UPT模型文档

### Dataset generation

```powershell
mkdir shapenet_car
cd shapenet_car
wget http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip
unzip mlcfd_data.zip
rm mlcfd_data.zip
rm -rf __MACOSX
cd mlcfd_data/training_data
tar -xvzf param0.tar.gz
tar -xvzf param1.tar.gz
tar -xvzf param2.tar.gz
tar -xvzf param3.tar.gz
tar -xvzf param4.tar.gz
tar -xvzf param5.tar.gz
tar -xvzf param6.tar.gz
tar -xvzf param7.tar.gz
tar -xvzf param8.tar.gz
rm param0.tar.gz
rm param1.tar.gz
rm param2.tar.gz
rm param3.tar.gz
rm param4.tar.gz
rm param5.tar.gz
rm param6.tar.gz
rm param7.tar.gz
rm param8.tar.gz
# remove folders without quadpress_smpl.vtk
rm -rf ./param2/854bb96a96a4d1b338acbabdc1252e2f
rm -rf ./param2/85bb9748c3836e566f81b21e2305c824
rm -rf ./param5/9ec13da6190ab1a3dd141480e2c154d3
rm -rf ./param8/c5079a5b8d59220bc3fb0d224baae2a
python data/shapenetcar/preprocess.py --src <SRC_FOLDER> --dst <DST_FOLDER>
```

### Environment Set Up

```powershell
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
python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Training

```powershell
cd /home/Userlist/guhexiang/baidu_98_CFD/PaddleCFD/examples/UPT
export CUDA_VISIBLE_DEVICES=2,3,4,6
/home/Userlist/guhexiang/miniconda3/bin/conda run -n ppcfd python main_train.py \
  --accelerator gpu \
  --devices 0,1,2,3 \
  --wandb_mode disabled \
  --hp yamls/shapenetcar/upt/dim768_seq1024sdf512_cnext_lr5e4_sd02_reprcnn_grn_grid32.yaml

```