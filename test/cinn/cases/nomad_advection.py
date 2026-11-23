"""NOMAD pure advection operator, nonlinear decoder, n=10 (examples/nomad/advection/train.py)."""
from cases import _common as C


EXAMPLE = "nomad/advection"
NOTE = "random ic/solution instead of pure_advection_traintest.npz; README setting `10 nonlinear`"
BATCH_SIZE = 100
M = 256
P = 25600
N = 10
DY = 2
DS = 1


def build_model():
    nomad = C.ppcfd_module("nomad.advection.operator_model")
    return nomad.OperatorModel(
        [M, 100, 100, 100, 100, 100, DS * N],
        [DS * N * 2, 100, 100, 100, 100, 100, DS],
        M,
        P,
        N,
        "nonlinear",
        DS,
    )


def build_batch():
    inputs = {"u": C.randn([BATCH_SIZE, M, 1]), "y": C.uniform([BATCH_SIZE, P, DY])}
    return inputs, {"s": C.randn([BATCH_SIZE, P, DS])}


def forward(model, inputs):
    return model((inputs["u"], inputs["y"]))


def compute_loss(model, inputs, labels):
    return model.loss(((inputs["u"], inputs["y"]), labels["s"]))
