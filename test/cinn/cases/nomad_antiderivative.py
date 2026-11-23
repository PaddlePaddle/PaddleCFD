"""NOMAD antiderivative operator, nonlinear decoder, n=10 (examples/nomad/antiderivative/train.py)."""
from cases import _common as C


EXAMPLE = "nomad/antiderivative"
NOTE = "random u/y/s instead of the generated sin(k x) data; README setting `10 nonlinear`"
BATCH_SIZE = 100
M = P = 500
N = 10
DS = 1


def build_model():
    nomad = C.ppcfd_module("nomad.antiderivative.nomad_antiderivative")
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
    inputs = {"u": C.randn([BATCH_SIZE, M, 1]), "y": C.uniform([BATCH_SIZE, P, 1])}
    return inputs, {"s": C.randn([BATCH_SIZE, P, DS])}


def forward(model, inputs):
    return model((inputs["u"], inputs["y"]))


def compute_loss(model, inputs, labels):
    return model.loss(((inputs["u"], inputs["y"]), labels["s"]))
