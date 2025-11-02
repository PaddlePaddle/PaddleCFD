# A time-series forecasting model for hydrodynamic loads of a ventilated cavitating body
***

## 1.Background
Under different operating conditions, the ventilated cavity on the surface of the axisymmetric body exhibits distinct morphological characteristics. When the trailing edge of the cavity breaks off or detaches, the lateral force (torque) experienced by the axisymmetric body shows strong pulsatility, which in turn affect its motion stability. In engineering applications, a closed-loop control system can be implemented to adjust the cavitaty morphology in real time; however, this is influenced by fluid hysteresis effects. Therefore, predicting the long-term lateral force (torque) of the axisymmetric body in advance is crucial for its stability control.
***
## 2.Model Description
Due to the constraints of installation space, in practical engineering applications, only operating condition information and pressure data from a limited number of points can be monitored, while information on forces over the past period is unavailable. This poses an inherent challenge for the aforementioned time-series prediction task. To address this, we developed an advanced TransKAN model to perform the time-series prediction, with its framework structured as follows.
![Structure of the TransKAN model](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/image/transkan.jpg)
It can be seen that the main body of the model is based on the Transolver framework, with a KAN module included before the output layer. Specifically, the Transolver employs a physical multi-head self-attention mechanism, as follows:
It first extracts raw features  and physical features  using two separate linear layers, as shown in Equations below.

$$
x_{\text{raw}} = \mathrm{linear}_x(x)
$$

$$
x_{\text{physical}} = \mathrm{linear}_{fx}(x)
$$

Next, the input sequence is aggregated into _S_ representative slices through a linear layer, and the correlation weights **_W_** between each time step and different slices are calculated .

$$
\boldsymbol{W} = \text{softmax}\left( \frac{\text{linear}_{{slice}}(x_{\text{raw}})}{\tau} \right)
$$

Then, the correlation weights **_W_** is used to perform spatially weighted aggregation on the physical features, resulting in a physics-aware token **_Z_**.

$$
\boldsymbol{Z} = \frac{\boldsymbol{W}^T x_{\text{physical}}}{\boldsymbol{W}^T I + 10^{-6}}
$$

Subsequently, the query **_q_**, key **_k_**, and value **_v_** are derived from the token **_Z_**  through three separate linear layers, followed by the computation of dot-product attention.

$$
\boldsymbol{q} = \text{linear}_q(\boldsymbol{Z}), \boldsymbol{k} = \text{linear}_k(\boldsymbol{Z}), \boldsymbol{v} = \text{linear}_v(\boldsymbol{Z})
$$

$$
\boldsymbol{Z'} = \text{softmax}\left( \frac{\boldsymbol{q}\boldsymbol{k}^T}{\sqrt{d}} \right) \boldsymbol{v}
$$

Finally, the attention output feature **_Z′_**  is reconstructed using the correlation weights **_W_** .

$$
\boldsymbol{Z'} = \boldsymbol{W}\boldsymbol{Z'}
$$

The KAN module is used to perform a nonlinear mapping along the temporal dimension of the input sequence, thereby allowing flexible configuration of the input and output sequence lengths. Its specific computational procedure is as follows.

$$
\text{KANlayer}^{(1)}(x) = \tanh\left( \text{Linear}^{(1)}_{\sigma}(\sigma(x)) + \text{Linear}^{(1)}_{B}(B(x)) \right)
$$

$$
\text{KANlayer}^{(2)}(x) = \text{Linear}^{(2)}_{\sigma}(\sigma(x)) + \text{Linear}^{(2)}_{B}(B(x))
$$

Here, ![formula](https://latex.codecogs.com/svg.image?\text{Linear}^{(i)}_{\sigma}) and ![formula](https://latex.codecogs.com/svg.image?\text{Linear}^{(i)}_{B}) are learnable linear combinations used to control the output dimension of each layer; 
$\sigma$ is the nonlinear activation function SiLU; _B_ is the B-spline function.
***
##  3.Dataset
本项目需要时序数据构造样本，在模型训练及测试之前需要对数据进行预处理，并保存为csv格式，且需要：
总共 _n_ ×5000行，其中 _n_ 为工况数，5000为每个工况的时间步数；共10列，其中1 ~ 3列为工况参数，4 ~ 8为5个迎、背水面测点压差，9、10列分别为侧向力和侧向力矩。
This project requires constructing time-series data samples. Before model training and testing, the data need to be preprocessed and saved in CSV format, with the following specifications:
 - A total of n × 5000 rows, where n is the number of operating conditions, and 5000 is the number of time steps for each condition.

 - A total of 10 columns: columns 1–3 correspond to the operating condition parameters, columns 4–8 correspond to the differential pressures at five measurement points on the pressure and suction surfaces, and columns 9 and 10 correspond to the lateral force and lateral moment, respectively.

The specific format can be found in[data_test.csv](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/data/data_test.csv)

本项目所构建的TransKAN模型可适用于解决多特征、多输出、多对多的时序预测问题，其中关于时序样本输入特征/输出标签以及输入/输出序列长度的选择位于[functiondata.py](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/functions_data.py)中的第118和119行，可以根据需要自行调整。
```
inputs = self.dataset_dict[self.mode][index, :self.t_len_in, :8]                                  
labels = self.dataset_dict[self.mode][index, self.t_len_in: self.t_len_in + self.t_len_out, 8:]   
```
***
## 4.Model Training & prediction

### (1) This project was developed and tested with **Python 3.10.16**.  
To ensure proper execution, please install the following Python packages with the specified versions:

```bash
pip install einops==0.8.1
pip install hydra-core==1.3.2
pip install matplotlib==3.10.7
pip install numpy==1.23.1
pip install omegaconf==2.3.0
pip install paddlepaddle-gpu==3.0.0b1
pip install paddlesci==1.3.0
pip install pandas==2.3.3
pip install scipy==1.16.3
```
Attention: Ensure your CUDA version is compatible with `paddlepaddle-gpu==3.0.0b1`
### (2) 设置config.yaml文件中的参数，具体包括：训练/测试模式选择、输入/输出序列长度设定、训练/测试集比例划分、工况数设定以及是否在训练中进行验证
```
# Setting
mode: "test"

# DataSet
ratio: [0.0, 0.0, 1.0]
t_len_in: 100
t_len_out: 50
num_conditions: 1

# eval
with_val: true
pred_ckpt: null
```

 -  Train or test the model:
```
python main.py
```
if run successfully:
![Successfully run](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/image/run.png)
***
## 5.Result
下图展示了TransKAN模型的通气空泡回转体未来长时段侧向力（矩）脉动量的预示结果与真实结果的比较，It can be seen that the predicted curve aligns well with the true curve in terms of overall pulsation frequency. Moreover, the phase and amplitude within each pulsation cycle are basically synchronized with the true curve. This indicates strong agreement between the predictions and true results, demonstrating the excellent temporal prediction capability of the TransKAN model.
![F](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/image/F.jpg) ![T](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/image/T.jpg) 
***
## 6.Reference
Y. Li, R. Huang, Y. Wang, L. Hao, and T. Liu, " TransKAN-GPR: A multi-scale framework for predicting hydrodynamic loads of a ventilated cavitating body,".(under review)
