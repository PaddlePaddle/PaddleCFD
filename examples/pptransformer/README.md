
enable_
The parallel efficiency achieved 90.2% in the data parallel computing experiment.


datadownload
- shapenet-car : https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/ShapeNet-Car/mlcfd_data.zip

- drivaernet++(downsample) : https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/drivaer_pp.tar


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
