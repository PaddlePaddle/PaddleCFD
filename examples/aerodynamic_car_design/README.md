This model can predict drag of the vehicle in different geometry
# 1. Model Traits:
## Speed Up

DataSet  | transolver original repo  | paddle 
-- | -- | -- 
ShapeNet-Car | 12 hours | 2 hours
DrivAerNet++ | TODO | TODO

- The parallel efficiency achieved 90.2% in the data parallel computing experiment.


# 2. Percision
- ShapeNet-Car

physics  | l2 transolver original repo  | l2 paddle 
-- | -- | -- 
surf | 0.0769  | 0.768
volume | 0.0211 | 0.0253

- DrivAerNet++ (TODO)


# 3. Enviroment
## 3.1 Linux: data and checkpoint download
``` Data
cd examples/aerodynamic_car_design/
mkdir -p ./data && cd ./data
wget https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip
unzip mlcfd_data.zip
cd ..
```

``` Checkpoint
mkdir -p ./checkpoint/shapenet_car && cd ./checkpoint/shapenet_car
wget https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/pptransformer/model_131.pdparams
cd .. && cd ..
```

## 3.2 Windows: data and checkpoint download

[click_me_to_download_data](https://paddle-org.bj.bcebos.com/paddlecfd/datasets/pptransformer/mlcfd_data.zip)

[click_me_to_download_checkpoint](https://paddle-org.bj.bcebos.com/paddlecfd/checkpoints/pptransformer/model_131.pdparams)

# 4. How to Run
## ShapeNet-Car
``` Train
python main_shapenetcar.py
```

``` Test
export PYTHONPATH=../../:${PYTHONPATH}
python main_shapenetcar.py \
    mode=test \
    checkpoint=./checkpoint/shapenet_car/model_131.pdparams
```

if test successfully:
<img width="1825" height="707" alt="image" src="https://github.com/user-attachments/assets/0e44b21a-e2ff-440e-9663-f0848d88a274" />


## DrivAerNet++ (TODO)
```sh
python -m paddle.distributed.launch --gpus=0,1 main_v2.py \
    --config-name transolver_drivaerpp.yaml \
    lr=0.0001 \
    enable_dp=true \
    batch_size=2 \
    data_module.n_train_num=32 \
    data_module.n_test_num=1 \
    num_epochs=10
```

# 5. reference 
[1] Wu H, Luo H, Wang H, et al. Transolver: A fast transformer solver for pdes on general geometries[J]. arXiv preprint arXiv:2402.02366, 2024.
