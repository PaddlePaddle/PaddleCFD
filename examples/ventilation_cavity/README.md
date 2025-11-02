# A time-series forecasting model for hydrodynamic loads of a ventilated cavitating body
***

## 1. Background
在不同工况下，回转体表面的通气空泡会呈现不同的形态特征。当空泡尾部呈断裂脱落特性时，回转体所受侧向力（矩）会表现出较强的脉动性，进而影响其运动稳定性。工程中可以引入闭环控制系统对空泡形态进行实时调整，但会收到流体迟滞效应的影响。因此，对回转体未来长时段侧向力（矩）进行超前预示，对回转体的稳定性控制十分重要。
***
## 2.Model Description
受装配空间的限制，实际工程中只能监测工况信息以及有限测点的压力信息，无法提供过去一段时间的受力信息，这对上述时序预测任务带来了本质困难。对此，我们构建了一个先进的TransKAN模型来开展时序预测任务，其框架结构如下。
![Structure of the TransKAN model](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/image/transkan.jpg)
可以看出，该模型主体为transolver框架，且其输出层之前包含一个KAN模块。其中，transolver采用物理多头自注意力机制，具体来说：
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

KAN模块用于对输入序列的时间维数进行非线性映射，从而实现输入/输出序列长度的自由配置，其具体的计算流程如下。

$$
\text{KANlayer}^{(1)}(x) = \tanh\left( \text{Linear}^{(1)}_{\sigma}(\sigma(x)) + \text{Linear}^{(1)}_{B}(B(x)) \right)
$$

$$
\text{KANlayer}^{(2)}(x) = \text{Linear}^{(2)}_{\sigma}(\sigma(x)) + \text{Linear}^{(2)}_{B}(B(x))
$$

Here, ![formula](https://latex.codecogs.com/svg.image?\text{Linear}^{(i)}_{\sigma}) 
and ![formula](https://latex.codecogs.com/svg.image?\text{Linear}^{(i)}_{B}) 
are learnable linear combinations used to control the output dimension of each layer; 
$\sigma$ is the nonlinear activation function SiLU; _B_ is the B-spline function.
***
##  3.Dataset
本项目需要时序数据构造样本，在模型训练及测试之前需要对数据进行预处理，并保存为csv格式，且需要：
总共 _n_ ×5000行，其中 _n_ 为工况数，5000为每个工况的时间步数；共10列，其中1 ~ 3列为工况参数，4 ~ 8为5个迎、背水面测点压差，9、10列分别为侧向力和侧向力矩。
具体格式参考[data_test.csv](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/data/data_test.csv)
***
## 4.Model Training & prediction

 - 设置config.yaml文件中的参数，具体包括：训练/测试模式选择、输入/输出序列长度设定、训练/测试集比例划分、工况数设定以及是否在训练中进行验证
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
## 5.Result
下图展示了TransKAN模型的通气空泡回转体未来长时段侧向力（矩）脉动量的预示结果与真实结果的比较，It can be seen that the predicted curve aligns well with the true curve in terms of overall pulsation frequency. Moreover, the phase and amplitude within each pulsation cycle are basically synchronized with the true curve. This indicates strong agreement between the predictions and true results, demonstrating the excellent temporal prediction capability of the TransKAN model.
![F](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/image/F.jpg) ![T](https://github.com/lypUCAS/PaddleCFD/blob/develop/examples/ventilation_cavity/image/T.jpg) 
## 6.Reference
