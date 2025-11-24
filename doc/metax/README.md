
## PaddleCFD Metax README
### 1.Register an account and log in on the giteeAi platform
https://ai.gitee.com/
### 2.购买算力, 点击 '立即租用'
![alt text](image.png)
### 3.selecting the PaddleCFD image, click Next to create an instance.
![alt text](image-1.png)
### 4.Once the instance is created, as shown below, click on Lab to enter the container.
![alt text](image-2.png)
### 5.clicking Jupyter Lab, you will enter this interface. First, select the terminal; this article chooses Terminal.
![alt text](image-3.png)
## Training Process
PaddleCFD Source Dir:
```
cd /opt/package/ppcfd/PaddleCFD
```
All Case Training Reference Documents :

[1、PaddleCFD aerodynamic_car_design pptransformer README](https://github.com/PaddlePaddle/PaddleCFD/blob/develop/examples/aerodynamic_car_design/README.md) 

[2、PaddleCFD aerodynamic_drag_pred ppfno README](https://github.com/PaddlePaddle/PaddleCFD/blob/develop/examples/aerodynamic_drag_pred/ppfno/README.md) 

[3、PaddleCFD aerodynamics ppkan README](https://github.com/PaddlePaddle/PaddleCFD/blob/develop/examples/aerodynamics/ppkan/README.md) 

[4、PaddleCFD airfoil_wake_nodm README](https://github.com/PaddlePaddle/PaddleCFD/blob/develop/examples/airfoil_wake/README.md) 

[5、PaddleCFD darcyflow ppdeeponet README](https://github.com/PaddlePaddle/PaddleCFD/blob/develop/examples/darcyflow/ppdeeponet/README.md) 

[6、PaddleCFD darcyflow ppkan README](https://github.com/PaddlePaddle/PaddleCFD/blob/develop/examples/darcyflow/ppkan/README.md) 

[7、PaddleCFD ppdiffusion README](https://github.com/PaddlePaddle/PaddleCFD/blob/develop/examples/flow_field_prediction/ppdiffusion/README.md)


Using the darcyflow ppkan case as an example to explain the training process:

### 1.Get the dataset
```sh
cd /data
wget https://paddle-org.bj.bcebos.com/paddlecfd/datasets/ppkan/piececonst_r421_N1024_smooth1.mat

cd /opt/package/ppcfd/PaddleCFD/examples/darcyflow/ppkan
ln -sf /data/piececonst_r421_N1024_smooth1.mat ./data/piececonst_r421_N1024_smooth1.mat
```

### 2.Train

```sh
python main.py mode=train
```
<img width="1603" height="448" alt="517989774-ae2e8176-679d-46ba-b0db-83fbd394c38c" src="https://github.com/user-attachments/assets/559b53d8-82d9-4f24-9f38-97939f4fa406" />

### 3.Eval

```python
python main.py mode=test checkpoint=./outputs-KANONet/2025-11-24/15-02-32/KANONet_latest.pdparams
```
<img width="1607" height="321" alt="517990409-58602ac5-f96a-4d66-82e5-4430e42962c0" src="https://github.com/user-attachments/assets/4cdc336f-c0a0-4219-b03c-13cacbae9688" />
 Results: The pressure field prediction relative err on the test set is MSE is 0.0071.
<img width="1010" height="485" alt="517992729-b4c8eb6b-8fb2-4200-b5f1-a97732344133" src="https://github.com/user-attachments/assets/21eadcf7-ebdc-4d01-9093-6d799b5099a7" />

 
