"""NOMAD shallow-water operator, nonlinear decoder, n=10 (examples/nomad/shallowwater/train.py)."""
from cases import _common as C


EXAMPLE = "nomad/shallowwater"
NOTE = "random u/y/s instead of train_SW.npz; README setting `10 nonlinear`"
BATCH_SIZE = 100
M = 1024
P = 128
N = 10
DU = DY = DS = 3


def build_model():
    nomad = C.ppcfd_module("nomad.shallowwater.nomad_model")
    return nomad.OperatorModel(
        [M * DU, 100, 100, 100, 100, 100, DS * N],
        [DS * N * 2, 100, 100, 100, 100, 100, DS],
        n=N,
        decoder="nonlinear",
        ds=DS,
    )


def build_batch():
    inputs = {"u": C.randn([BATCH_SIZE, M, DU]), "y": C.uniform([BATCH_SIZE, P, DY])}
    return inputs, {"s": C.randn([BATCH_SIZE, P, DS])}


def forward(model, inputs):
    return model((inputs["u"], inputs["y"]))


def compute_loss(model, inputs, labels):
    return model.loss(((inputs["u"], inputs["y"]), labels["s"]))
