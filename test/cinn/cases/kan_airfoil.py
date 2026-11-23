"""B-spline KAN DeepONet (KANONet) on AirfRANS (examples/aerodynamics/ppkan/conf/main.yaml)."""
from cases import _common as C


EXAMPLE = "aerodynamics/ppkan"
NOTE = "synthetic branch [2000, 63] in [-1, 1] / trunk [2000, 2] in [-1.5, 3] instead of AirfRANS"
BATCH_SIZE = 2000


def build_model():
    kan = C.ppcfd_module("ppkan.kan")
    return kan.KANONet(
        width_branch1=[63, 128, 128, 128],
        grid_range_b1=[-1.0, 1.0],
        grid_size_b1=10,
        width_trunk=[2, 64, 128, 128],
        grid_range_t=[-1.5, 3.0],
        grid_size_t=10,
    )


def build_batch():
    inputs = {
        "branch1": C.uniform([BATCH_SIZE, 63], -1.0, 1.0),
        "trunk": C.uniform([BATCH_SIZE, 2], -1.5, 3.0),
    }
    return inputs, {"y": C.randn([BATCH_SIZE, 4])}


def compute_loss(model, inputs, labels):
    return C.mse(model(inputs), labels["y"])
