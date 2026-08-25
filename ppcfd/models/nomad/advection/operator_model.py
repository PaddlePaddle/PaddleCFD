import numpy as np
import paddle
import paddle.nn as nn

from .mlp import MLP


class OperatorModel(nn.Layer):

    def __init__(
        self,
        branch_layers,
        trunk_layers,
        m=100,
        P=100,
        n=None,
        decoder="nonlinear",
        ds=1,
    ):
        super().__init__()

        self.branch_net = MLP(branch_layers)
        self.trunk_net = MLP(trunk_layers)

        self.n = n
        self.ds = ds
        self.m = m
        self.P = P
        self.decoder = decoder

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
            t, [inputsy.shape[0], inputsy.shape[1], self.ds, self.n]
        )

        B = inputsu.shape[0]

        b = paddle.reshape(inputsu, [B, 1, inputsu.shape[1] * inputsu.shape[2]])

        b = self.branch_net(b)

        b = paddle.reshape(
            b, [b.shape[0], int(b.shape[2] / self.ds), self.ds]
        )

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

        loss = paddle.mean(
            (paddle.flatten(y) - paddle.flatten(y_pred)) ** 2
        )

        return loss

    def L2error(self, batch):

        inputs, y = batch

        y_pred = self.forward(inputs)

        num = paddle.norm(paddle.flatten(y) - paddle.flatten(y_pred), p=2)

        den = paddle.norm(paddle.flatten(y), p=2)

        return num / den

    def predict(self, inputs):

        self.eval()

        with paddle.no_grad():
            s_pred = self.forward(inputs)

        return s_pred

    def count_params(self):

        total = sum(np.prod(p.shape) for p in self.parameters())

        print("Total parameters:", int(total))