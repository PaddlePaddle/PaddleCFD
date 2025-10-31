import paddle
from paddle.nn import initializer as I
import math
from einops import rearrange

ACTIVATION = {
    "gelu": paddle.nn.GELU,
    "tanh": paddle.nn.Tanh,
    "sigmoid": paddle.nn.Sigmoid,
    "relu": paddle.nn.ReLU,
    "leaky_relu": paddle.nn.LeakyReLU(negative_slope=0.1),
    "softplus": paddle.nn.Softplus,
    "ELU": paddle.nn.ELU,
    "silu": paddle.nn.Silu,
}


### Z-score归一化（反归一化） ###
class EmbInstanceNorm(paddle.nn.Layer):
    def __init__(self, epsilon=1e-5):
        super(EmbInstanceNorm, self).__init__()
        self.epsilon = epsilon # 数值稳定参数，防止分母为0的情况发生
        self.mean = paddle.to_tensor([[[1, 100]]], dtype=paddle.get_default_dtype()) # 这里对均值和方差初始化赋值
        self.var = paddle.to_tensor([[[1, 10]]], dtype=paddle.get_default_dtype())

    # 归一化（每个样本，每个维度上的所有时间步的取值进行归一化，均值和方差依赖于样本本身）
    def forward(self, x): # x的形状 = [批大小,时间步数,维数]
        mean = x.mean(axis=1, keepdim=True)
        variance = ((x - mean) ** 2).mean(axis=1, keepdim=True)
        normalized = (x - mean) / paddle.sqrt(variance + self.epsilon)
        return normalized

    # 反归一化（均值和方差依赖于初始化赋值），这里存在问题，后续需要修改！！！
    def denorm(self, x):
        denormalized = x * paddle.sqrt(self.var + self.epsilon) + self.mean
        return denormalized


### 基于物理感知的多头自注意力层 ###
# 输入一个batch_size的样本 x ，对于其中的每个样本，输出提取出来的时序关系
class Physics_Attention(paddle.nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64): # 要根据实际数据调整 slice_num（要小于时间步数）
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5 # 缩放因子（防止点积后的数值过大/过小），这与transformer中的attention机制的处理方式一致
        self.softmax = paddle.nn.Softmax(axis=-1) # 对注意力分数做Softmax
        self.dropout = paddle.nn.Dropout(p=dropout)
        # 温度参数（物理注意力机制的特殊所在）
        temperature = paddle.ones(shape=[1, heads, 1, 1]) * 0.5 # 形状为（1, num_heads, 1, 1）
        self.temperature = paddle.create_parameter(
            shape=temperature.shape,
            dtype=temperature.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(temperature),
        )
        self.temperature.stop_gradient = False
        # 定义各种计算矩阵
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=inner_dim)  # 原始特征提取
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features=inner_dim) # 物理特征提取
        self.in_project_slice = paddle.nn.Linear(in_features=dim_head, out_features=slice_num)
        for la in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(la.weight)
        # qkv计算，不改变维度
        self.to_q = paddle.nn.Linear(in_features=dim_head, out_features=dim_head, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=dim_head, out_features=dim_head, bias_attr=False)
        self.to_v = paddle.nn.Linear( in_features=dim_head, out_features=dim_head, bias_attr=False)
        self.to_out = paddle.nn.Sequential( # 相当于dense层
            paddle.nn.Linear(in_features=inner_dim, out_features=dim),
            paddle.nn.Dropout(p=dropout),
        )

    def forward(self, x):
        B, N, C = tuple(x.shape)  # 提取输入样本 x 的批大小、时间步数和维数
        # 得到每个头所提取的x的物理特征
        fx_mid = (
            self.in_project_fx(x)
            .reshape([B, N, self.heads, self.dim_head])
            .transpose(perm=[0, 2, 1, 3]) # 由于多头，每个头提取的特征彼此独立，这里要调整数据结构（方面存取）
        )
        # 得到每个头所提取的x的原始特征
        x_mid = (
            self.in_project_x(x)
            .reshape([B, N, self.heads, self.dim_head])
            .transpose(perm=[0, 2, 1, 3])
        )
        # 物理聚类
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature) # 使用广播规则，将分母形状调整为分子的，从而最终结果形状为分子形状
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) #爱因斯坦求和规则
        slice_token = slice_token / (slice_norm + 1e-05).unsqueeze(-1).expand([-1, -1, -1, self.dim_head]) # unsqueeze在最后插入一个维度
        # 计算qkv                                                                                          # expand扩展维数，-1为保持，16为目标
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        # 计算c
        x = k_slice_token
        perm_2 = list(range(x.ndim))
        perm_2[-1] = -2
        perm_2[-2] = -1                        # 交换最后两个维度
        dots = paddle.matmul(x=q_slice_token, y=x.transpose(perm=perm_2)) * self.scale # 矩阵乘法
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)
        # 输出
        out_x = paddle.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


### 带残差连接的多层感知机 ###
class MLP(paddle.nn.Layer):
    def __init__(
        # 输入维数、隐藏层维数、输出维数、隐藏层数、激活函数、启用残差连接
        self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True, dropout=0.0
    ):
        super(MLP, self).__init__()
        # 检查激活函数可用性
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        # 输入层映射到隐藏层
        self.linear_pre = paddle.nn.Sequential(paddle.nn.Linear(in_features=n_input, out_features=n_hidden), act())
        # 隐藏层映射到输出层（可以结合输入、输出维数识别）
        self.linear_post = paddle.nn.Linear(in_features=n_hidden, out_features=n_output)
        # 隐藏层映射到隐藏层
        self.linears = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Sequential(
                    paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden), # 这里输入、输出维数相同是为了便于残差连接
                    act(),
                    paddle.nn.Dropout(p=dropout),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x): # 输入 x 的形状（批大小、时间步数、维数）
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res: # 判断是否需要进行残差连接，作用是为了防止梯度消失
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x # 输出仅有维数改变


### 基本block ###
class Transolver_block(paddle.nn.Layer):
    def __init__(
        self,
        # 指定默认值
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4, # MLP隐藏层维数的扩展系数
        last_layer=False,
        out_dim=1,   # last_layer = true 时才生效
        slice_num=32,
    ):
        super().__init__()
        # 声明所调用的子块（包括自己编写和库中已有的）
        self.last_layer = last_layer
        self.ln_1 = paddle.nn.LayerNorm(normalized_shape=hidden_dim) # 归一化 不改变维数
        self.Attn = Physics_Attention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.ln_2 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=3, # 3个隐藏层
            res=False,  # 不启动残差连接
            act=act,
            dropout=dropout,
        )
        # 判断每个子块的输出是否为最后一层，从而调整输出维数
        if self.last_layer:
            self.ln_3 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
            self.mlp2 = paddle.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx # 先归一化，再计算自注意力，然后进行残差连接
        fx = self.mlp(self.ln_2(fx)) + fx  # 再次归一化，之后mpl，然后残差连接
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class KANLinear(paddle.nn.Layer):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=paddle.nn.Silu,
            grid_eps=0.02,
            grid_range=[-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
                paddle.arange(start=-spline_order, end=grid_size + spline_order + 1) * h
                + grid_range[0]
        ).expand(shape=[in_features, -1]).contiguous()
        self.register_buffer(name='grid', tensor=grid)

        self.base_weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[out_features, in_features]))
        self.spline_weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[out_features, in_features, grid_size + spline_order]))

        if enable_standalone_scale_spline:
            self.spline_scaler = (paddle.base.framework.EagerParamBase.
                                  from_tensor(tensor=paddle.empty(shape=[out_features,
                                                                         in_features])))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
            negative_slope=math.sqrt(5) * self.scale_base, nonlinearity=
            'leaky_relu')
        init_KaimingUniform(self.base_weight)
        with paddle.no_grad():
            noise = (
                            paddle.rand(shape=[self.grid_size + 1, self.in_features, self.out_features]) - 1 / 2
                    ) * self.scale_noise / self.grid_size

            paddle.assign(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise),
                output=self.spline_weight.data)

            if self.enable_standalone_scale_spline:
                init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5) * self.scale_spline,
                    nonlinearity='leaky_relu')
                init_KaimingUniform(self.spline_scaler)

    def b_splines(self, x: paddle.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            paddle.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.shape[1] == self.in_features
        grid: paddle.Tensor = self.grid
        x = x.unsqueeze(axis=-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] \
                    + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]

        assert tuple(bases.shape) == (
            x.shape[0],
            self.in_features,
            self.grid_size + self.spline_order)

        return bases.contiguous()

    def curve2coeff(self, x: paddle.Tensor, y: paddle.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).
            y (paddle.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            paddle.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.shape[1] == self.in_features
        assert tuple(y.shape) == (x.shape[0], self.in_features, self.out_features)

        A = self.b_splines(x).transpose(perm=dim2perm(self.b_splines(x).
                                                      ndim, 0,
                                                      1))  # [in_features, batch_size, grid_size + spline_order]
        B = y.transpose(perm=dim2perm(y.ndim, 0, 1))  # [in_features, batch_size, out_features]
        solution = paddle.linalg.lstsq(x=A, y=B)[0]  # [in_features, grid_size + spline_order, out_features]
        if A.shape[0] == 1:
            solution = solution.unsqueeze(axis=0)
        # print("A shape: ", A.shape, "B shape: ", B.shape, "Solution shape: ", solution.shape)
        result = solution.transpose([2, 0, 1])
        assert tuple(result.shape) == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order)

        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(axis=-1)
            if self.enable_standalone_scale_spline
            else 1.0)

    def forward(self, x: paddle.Tensor):
        assert x.dim() == 2 and x.shape[1] == self.in_features

        base_output = paddle.nn.functional.linear(
            x=self.base_activation(x),
            weight=self.base_weight.T)

        spline_output = paddle.nn.functional.linear(
            x=self.b_splines(x).reshape([x.shape[0], -1]).contiguous(),  # .view(x.shape[0], -1),
            weight=self.scaled_spline_weight.reshape(
                [self.out_features, -1]).T.contiguous())  # .view(self.out_features, -1).T)
        # paddle.view leads to error
        # spline_output = paddle.nn.functional.linear(
        #     x=self.b_splines(x).view(x.shape[0], -1),
        #     weight=self.scaled_spline_weight.view(self.out_features, -1).T)

        return base_output + spline_output

    @paddle.no_grad()
    def update_grid(self, x: paddle.Tensor, margin=0.01):
        assert x.dim() == 2 and x.shape[1] == self.in_features
        batch = x.shape[0]

        splines = self.b_splines(x)  # [batch, in, coeff]
        splines = splines.transpose(perm=[1, 0, 2])  # [in, batch, coeff]
        orig_coeff = self.scaled_spline_weight  # [out, in, coeff]
        orig_coeff = orig_coeff.transpose(perm=[1, 2, 0])  # [in, coeff, out]
        unreduced_spline_output = paddle.bmm(x=splines, y=orig_coeff)  # [in, batch, out]
        unreduced_spline_output = unreduced_spline_output.transpose(perm=[1, 0, 2])  # [batch, in, out]

        # sort each channel individually to collect data distribution
        x_sorted = (paddle.sort(x=x, axis=0), paddle.argsort(x=x, axis=0))[0]
        grid_adaptive = x_sorted[
            paddle.linspace(start=0, stop=batch - 1,
                            num=self.grid_size + 1, dtype='int64')
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = paddle.arange(
            dtype='float32', end=self.grid_size + 1
        ).unsqueeze(axis=1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = paddle.concat(x=[grid[:1] - uniform_step * paddle.arange(start=
                                                                        self.spline_order, end=0, step=-1).unsqueeze(
            axis=1), grid, grid[-1
                                :] + uniform_step * paddle.arange(start=1, end=self.spline_order +
                                                                               1).unsqueeze(axis=1)], axis=0)

        paddle.assign(grid.T, output=self.grid)
        paddle.assign(self.curve2coeff(x, unreduced_spline_output), output=self
                      .spline_weight.data)

    def regularization_loss(self, regularize_activation=1.0,
                            regularize_entropy=1.0):
        """
        Compute the regularization loss.

        L1 and the entropy loss is for the feature selection, i.e., let the weight of the activation function be small.
        """
        l1_fake = self.spline_weight.abs().mean(axis=-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -paddle.sum(x=p * p.log())
        return (regularize_activation * regularization_loss_activation +
                regularize_entropy * regularization_loss_entropy)


class KAN(paddle.nn.Layer):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=paddle.nn.Silu,
            grid_eps=0.02,
            grid_range=[-1, 1]
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = paddle.nn.LayerList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range))

    def forward(self, x: paddle.Tensor, update_grid=False):
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers) - 1:
                x = paddle.nn.functional.tanh(x=x)
        return x

    def regularization_loss(self, regularize_activation=1.0,
                            regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation,
                                             regularize_entropy) for layer in self.layers)


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


### 构建transKAN模型 ###
class Transolver(paddle.nn.Layer):
    def __init__(
        self,
        # 指定默认值，这里解释含义，具体取值参考config.yaml
        space_dim=8,   # 输入序列维数
        n_layers=5,    # 基本block数
        n_hidden=64,   # 隐藏层维数（适用于多头物理自注意力模块和MLP模块）
        dropout=0.15,  # 丢弃率
        n_head=8,      # 多头物理自注意力模块的头数
        act="gelu",    # 激活函数，用于MLP
        mlp_ratio=1,   # MLP的扩展系数
        fun_dim=0,     # 是否使用时间戳
        out_dim=1,     # 输出序列维数
        slice_num=32,  # 切片数，用于多头物理自注意力模块
        **kwargs,
    ):
        super(Transolver, self).__init__()
        self.n_layers = n_layers
        # 预处理MLP
        self.preprocess = MLP(
            space_dim + fun_dim ,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        # 拼接多个基本block
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=(i == n_layers - 1), # 用于判断block是否为最后一个
                )
                for i in range(n_layers)
            ]
        )
        # 定义时间信息的可学习替代
        param = 1 / n_hidden * paddle.rand(shape=(n_hidden,), dtype="float32")
        self.placeholder = paddle.create_parameter(
            shape=param.shape,
            dtype=param.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(param),
        )
        self.placeholder.stop_gradient = False

        #定义KAN
        self.apply_kan = kwargs.get("apply_kan", False)
        self.kan_hidden_layer = kwargs.get("kan_hidden_layer", [1, 5, 1])
        self.kan_grid_size = kwargs.get("kan_grid_size", 10)
        self.kan_grid_range = kwargs.get("kan_grid_range", [-1, 1])
        self.t_len_out = kwargs.get("t_len_out", 1)
        if self.apply_kan:
            self.postkan = KAN(
                layers_hidden=self.kan_hidden_layer,
                grid_size=self.kan_grid_size,
                spline_order=3,
                scale_noise=0.05,
                scale_base=1.0,
                scale_spline=1.0,
                grid_range=self.kan_grid_range
                # spline function domain, determined by the min/max of the predictions domain
            )

    # 权重初始化方法（针对paddle.nn.Linear和paddle.nn.LayerNorm，因为只有这两个里面包含可学习参数）
        self.initialize_weights()
    def initialize_weights(self):
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal = I.TruncatedNormal(std=0.02)
            trunc_normal(m.weight)  # 初始化权重
            if m.bias is not None:
                I.Constant(0.0)(m.bias)  # 偏置设为0
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm1D)):
            if m.bias is not None:
                I.Constant(0.0)(m.bias)
            if m.weight is not None:
                I.Constant(1.0)(m.weight)

    # 误差反向传播
    def forward(self, x, t=None):
        if t is not None:
            fx = paddle.concat([t, x], axis=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
            # print(paddle.min(fx).item(), paddle.max(fx).item(),paddle.mean(fx, axis=[0,1]).numpy())
        if self.apply_kan:
            batch = fx.shape[0]
            channels = fx.shape[1]
            features = fx.shape[2]
            fx = paddle.transpose(fx, perm=[0, 2, 1])
            fx = paddle.reshape(fx, shape=[-1,
                                           channels])  # channels should be equal to kan input features (hidden_layer[0])
            fx = self.postkan(fx)
            fx = paddle.reshape(fx, shape=[batch, features, self.t_len_out])
            fx = paddle.transpose(fx, perm=[0, 2, 1])
        else:
            fx = paddle.mean(fx, axis=1, keepdim=True)  # avg pool
        return fx
