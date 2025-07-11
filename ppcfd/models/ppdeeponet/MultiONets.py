import numpy as np
import paddle


try:
    from FunActivation import FunActivation
except ImportError:
    from .FunActivation import FunActivation


class MultiONetBatch(paddle.nn.Layer):

    def __init__(
        self,
        in_size_x: int,
        in_size_a: int,
        hidden_list: list[int],
        activation_x="ReLU",
        activation_a="Tanh",
        dtype=None,
    ):
        super(MultiONetBatch, self).__init__()
        self.hidden_list = hidden_list
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
            1 / in_size_x
        weight_attr_x = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / in_size_x), high=np.sqrt(1 / in_size_x))
        )
        bias_attr_x = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / in_size_x), high=np.sqrt(1 / in_size_x))
        )
        weight_attr_a = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / in_size_a), high=np.sqrt(1 / in_size_a))
        )
        bias_attr_a = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(low=-np.sqrt(1 / in_size_a), high=np.sqrt(1 / in_size_a))
        )
        self.fc_x_in = paddle.nn.Linear(
            in_features=in_size_x, out_features=hidden_list[0], weight_attr=weight_attr_x, bias_attr=bias_attr_x
        )
        self.fc_a_in = paddle.nn.Linear(
            in_features=in_size_a, out_features=hidden_list[0], weight_attr=weight_attr_a, bias_attr=bias_attr_a
        )
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(
                low=-np.sqrt(1 / self.hidden_in), high=np.sqrt(1 / self.hidden_in)
            )
        )
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(
                low=-np.sqrt(1 / self.hidden_in), high=np.sqrt(1 / self.hidden_in)
            )
        )
        for hidden in hidden_list:
            net_x.append(
                paddle.nn.Linear(
                    in_features=self.hidden_in, out_features=hidden, weight_attr=weight_attr, bias_attr=bias_attr
                )
            )
            net_a.append(
                paddle.nn.Linear(
                    in_features=self.hidden_in, out_features=hidden, weight_attr=weight_attr, bias_attr=bias_attr
                )
            )
            self.hidden_in = hidden
        self.net_x = paddle.nn.Sequential(*net_x)
        self.net_a = paddle.nn.Sequential(*net_a)
        self.w = paddle.nn.ParameterList(
            [
                paddle.create_parameter(
                    shape=[], dtype="float32", default_initializer=paddle.nn.initializer.Constant(1.0)
                )
                for _ in range(len(hidden_list))
            ]
        )
        self.b = paddle.nn.ParameterList(
            [
                paddle.create_parameter(
                    shape=[], dtype="float32", default_initializer=paddle.nn.initializer.Constant(0.0)
                )
                for _ in range(len(hidden_list))
            ]
        )

    def forward(self, x, a_mesh):
        """
        Input:
            x: size(n_batch, n_mesh, dx)
            a_mesh: size(n_batch, latent_size)
        """
        assert tuple(x.shape)[0] == tuple(a_mesh.shape)[0]
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        out = 0.0
        for net_x, net_a, w, b in zip(self.net_x, self.net_a, self.w, self.b):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += (x * a_mesh.unsqueeze(1)).sum(axis=2) * w + b
        out = out / len(self.net_x)
        return out.unsqueeze(axis=-1)


class MultiONetBatch_X(paddle.nn.Layer):
    """Multi-Input&Output case"""

    def __init__(
        self,
        in_size_x: int,
        in_size_a: int,
        latent_size: int,
        out_size: int,
        hidden_list: list[int],
        activation_x="ReLU",
        activation_a="Tanh",
        dtype=None,
    ):
        super(MultiONetBatch_X, self).__init__()
        self.hidden_list = hidden_list
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
        self.fc_x_in = paddle.nn.Linear(in_features=in_size_x, out_features=hidden_list[0])
        self.fc_a_in = paddle.nn.Linear(in_features=in_size_a, out_features=hidden_list[0])
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            net_x.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=hidden))
            net_a.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=hidden))
            self.hidden_in = hidden
        self.net_x = paddle.nn.Sequential(*net_x)
        self.net_a = paddle.nn.Sequential(*net_a)
        self.fc_out = paddle.nn.Linear(in_features=latent_size, out_features=out_size)

    def forward(self, x, a_mesh):
        """
        Input:
            x: size(n_batch, n_mesh, dx)
            a_mesh: size(n_batch, latent_size, da)
        """
        assert tuple(x.shape)[0] == tuple(a_mesh.shape)[0]
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        out = 0.0
        for net_x, net_a in zip(self.net_x, self.net_a):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += paddle.matmul(x, a_mesh.transpose([0, 2, 1]))
        out = self.fc_out(out / len(self.net_x))
        return out


class MultiONetCartesianProd(paddle.nn.Layer):

    def __init__(
        self,
        in_size_x: int,
        in_size_a: int,
        hidden_list: list[int],
        activation_x="ReLU",
        activation_a="Tanh",
        dtype=None,
    ):
        super(MultiONetCartesianProd, self).__init__()
        self.hidden_list = hidden_list
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
        self.fc_x_in = paddle.nn.Linear(in_features=in_size_x, out_features=hidden_list[0])
        self.fc_a_in = paddle.nn.Linear(in_features=in_size_a, out_features=hidden_list[0])
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            net_x.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=hidden))
            net_a.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=hidden))
            self.hidden_in = hidden
        self.net_x = paddle.nn.Sequential(*net_x)
        self.net_a = paddle.nn.Sequential(*net_a)
        self.w = paddle.nn.ParameterList(
            parameters=[
                paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.to_tensor(data=1.0))
                for _ in range(len(hidden_list))
            ]
        )
        self.b = paddle.nn.ParameterList(
            parameters=[
                paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.to_tensor(data=0.0))
                for _ in range(len(hidden_list))
            ]
        )

    def forward(self, x, a_mesh):
        """
        Input:
            x: size(mesh_size, dx)
            a_mesh: size(n_batch, latent_size)
        """
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        out = 0.0
        for net_x, net_a, w, b in zip(self.net_x, self.net_a, self.w, self.b):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += paddle.einsum("bh,mh->bm", a_mesh, x) * w + b
        out = out / len(self.net_x)
        return out.unsqueeze(axis=-1)


class MultiONetCartesianProd_X(paddle.nn.Layer):
    """Multi-Input&Output case"""

    def __init__(
        self,
        in_size_x: int,
        in_size_a: int,
        latent_size: int,
        out_size: int,
        hidden_list: list[int],
        activation_x="ReLU",
        activation_a="Tanh",
        dtype=None,
    ):
        super(MultiONetCartesianProd_X, self).__init__()
        self.hidden_list = hidden_list
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
        self.fc_x_in = paddle.nn.Linear(in_features=in_size_x, out_features=hidden_list[0])
        self.fc_a_in = paddle.nn.Linear(in_features=in_size_a, out_features=hidden_list[0])
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            net_x.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=hidden))
            net_a.append(paddle.nn.Linear(in_features=self.hidden_in, out_features=hidden))
            self.hidden_in = hidden
        self.net_x = paddle.nn.Sequential(*net_x)
        self.net_a = paddle.nn.Sequential(*net_a)
        self.fc_out = paddle.nn.Linear(in_features=latent_size, out_features=out_size)

    def forward(self, x, a_mesh):
        """
        Input:
            x: size(mesh_size, dx)
            a_mesh: size(n_batch, latent_size, da)
        """
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        out = 0.0
        for net_x, net_a in zip(self.net_x, self.net_a):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += paddle.einsum("bmh,nh->bnm", a_mesh, x)
        out = self.fc_out(out / len(self.net_x))
        return out
