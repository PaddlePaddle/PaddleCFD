"""Physics-informed MultiONet (piratenet) on piecewise-constant Darcy flow
(examples/darcyflow/ppdeeponet/pimultionet.py -c config_pwc.yaml)."""
import numpy as np
import paddle

from cases import _common as C


EXAMPLE = "darcyflow/ppdeeponet"
NOTE = (
    "synthetic piecewise-constant a in {5, 10} on the 29x29 mesh instead of smh/pwc .mat "
    "files; loss is the example's FDM residual of -div(a grad u) = 10 with mse_org "
    "(w_data = 0); harness uses Adam, the example SOAP + StepLR"
)
BATCH_SIZE = 50
RES = 29
LEARNING_RATE = 1e-3


class PIMultiONet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        encoder = C.ppcfd_module("ppdeeponet.EncoderNet")
        multionets = C.ppcfd_module("ppdeeponet.MultiONets")
        self.enc = encoder.EncoderCNNet2d(
            conv_arch=[1, 32, 64, 128],
            fc_arch=[128, 128],
            activation_conv="SiLU",
            activation_fc="SiLU",
            nx_size=RES,
            ny_size=RES,
            kernel_size=(5, 5),
            stride=2,
            dtype="float32",
        )
        self.u = multionets.MultiONetBatch_piratenet(
            in_size_x=2,
            in_size_a=128,
            hidden_list=[128, 128, 128],
            activation_x="ReLU",
            activation_a="SiLU",
            dtype="float32",
        )

    def forward(self, x, a):
        u = self.u(x, self.enc(a))
        return u * paddle.sin(np.pi * x[..., 0:1]) * paddle.sin(np.pi * x[..., 1:2])  # mollifier


def build_model():
    return PIMultiONet()


def _mesh():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, RES), np.linspace(0.0, 1.0, RES))
    return paddle.to_tensor(np.stack([x.flatten(), y.flatten()], 1), dtype="float32")


def build_batch():
    a = paddle.where(C.uniform([BATCH_SIZE, RES * RES, 1]) > 0.5, 10.0, 5.0)
    return {"a": a, "x_mesh": _mesh()}, {}


def forward(model, inputs):
    x = inputs["x_mesh"].tile([inputs["a"].shape[0], 1, 1])
    return model(x, inputs["a"])


def _fdm(u, delta):
    dudx = (u[:, 1:-1, 2:, :] - u[:, 1:-1, :-2, :]) / (2.0 * delta)
    dudy = (u[:, 2:, 1:-1, :] - u[:, :-2, 1:-1, :]) / (2.0 * delta)
    return dudx, dudy


def _fun_a(x, a):
    # pimultionet.fun_a: nearest-grid lookup of a at the points x
    delta = 1.0 / (RES - 1)
    a = paddle.squeeze(a, axis=-1)
    x_loc = paddle.floor(x[..., 0] / delta + 0.5).astype("int64")
    y_loc = paddle.floor(x[..., 1] / delta + 0.5).astype("int64")
    return a[paddle.arange(a.shape[0]).unsqueeze(1), y_loc * RES + x_loc].unsqueeze(-1)


def compute_loss(model, inputs, labels):
    n_batch = inputs["a"].shape[0]
    delta = 1.0 / (RES - 1)
    a = _fun_a(inputs["x_mesh"], inputs["a"]).reshape([-1, RES, RES, 1])
    u = forward(model, inputs).reshape([-1, RES, RES, 1])
    dudx, dudy = _fdm(u, delta)
    dauxdx, _ = _fdm(a[:, 1:-1, 1:-1, 0:1] * dudx, delta)
    _, dauydy = _fdm(a[:, 1:-1, 1:-1, 0:1] * dudy, delta)
    left = (-(dauxdx + dauydy)).reshape([n_batch, -1])
    right = 10.0 * paddle.ones_like(left)
    return paddle.mean(paddle.linalg.norm(right - left, p=2, axis=1))  # MyLoss.mse_org
