import numpy as np
import paddle
import paddle.nn as nn


class MLP(nn.Layer):
    def __init__(self, layers):
        super().__init__()
        layer_list = []
        for i in range(len(layers) - 2):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            layer_list.append(nn.GELU())

        layer_list.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):

        orig_shape = x.shape
        x = paddle.reshape(x, [-1, orig_shape[-1]])

        x = self.net(x)

        out_dim = x.shape[-1]

        x = paddle.reshape(x, list(orig_shape[:-1]) + [out_dim])

        return x


class OperatorModel(nn.Layer):

    def __init__(self,
                 branch_layers,
                 trunk_layers,
                 m=100,
                 P=100,
                 n=None,
                 decoder=None,
                 ds=None):

        super().__init__()

        self.branch_net = MLP(branch_layers)
        self.trunk_net = MLP(trunk_layers)

        self.n = n
        self.ds = ds
        self.m = m
        self.P = P
        self.decoder = decoder

        if decoder not in ["linear", "nonlinear"]:
            raise ValueError("decoder must be 'linear' or 'nonlinear'")

    def NOMAD(self, inputs):

        inputsu, inputsy = inputs

        B = inputsu.shape[0]

        b = paddle.reshape(inputsu, [B, 1, inputsu.shape[1]])
        b = self.branch_net(b)

        b = paddle.tile(b, repeat_times=[1, inputsy.shape[1], 1])

        repeat_factor = b.shape[-1] // inputsy.shape[-1]

        y_tiled = paddle.tile(inputsy, repeat_times=[1, 1, repeat_factor])

        inputs_recon = paddle.concat([y_tiled, b], axis=-1)

        out = self.trunk_net(inputs_recon)

        return out

    def DeepONet(self, inputs):

        inputsu, inputsy = inputs

        t = self.trunk_net(inputsy)

        t = paddle.reshape(
            t, [inputsy.shape[0], inputsy.shape[1], self.ds, self.n])

        b = paddle.reshape(
            inputsu, [inputsu.shape[0], 1, inputsu.shape[1] * inputsu.shape[2]])

        b = self.branch_net(b)

        b = paddle.reshape(b, [b.shape[0], int(b.shape[2] / self.ds), self.ds])

        Guy = paddle.einsum("ijkl,ilk->ijk", t, b)

        return Guy

    def forward(self, inputs):

        if self.decoder == "nonlinear":
            return self.NOMAD(inputs)

        elif self.decoder == "linear":
            return self.DeepONet(inputs)

    def loss(self, batch):

        inputs, y = batch

        y_pred = self.forward(inputs)

        loss = paddle.mean((paddle.flatten(y) - paddle.flatten(y_pred)) ** 2)

        return loss

    def L2error(self, batch):

        inputs, y = batch

        y_pred = self.forward(inputs)

        return paddle.norm(paddle.flatten(y) - paddle.flatten(y_pred), p=2) \
               / paddle.norm(paddle.flatten(y), p=2)

    def predict(self, inputs):

        self.eval()

        with paddle.no_grad():
            s_pred = self.forward(inputs)

        return s_pred

    def count_params(self):

        total = sum(np.prod(p.shape) for p in self.parameters())

        print("The number of model parameters is:", int(total))