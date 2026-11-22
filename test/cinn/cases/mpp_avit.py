"""MPP AViT-Ti on 2D shallow water (examples/multiple_physics_pretraining/config/mpp_avit_ti_config_cinn.yaml)."""
from types import SimpleNamespace

import paddle

from cases import _common as C


EXAMPLE = "multiple_physics_pretraining"
NOTE = (
    "synthetic 16-step history [16, 8, 1, 128, 128] of one field (state label 0, "
    "non-periodic bcs) instead of PDEBench shallow-water; DropPath (0.2 linspace) is set "
    "to 0 like the harness does for dropout; no pretrained weights; cinn config "
    "batch_size 8 / accum_grad 1 / Adam lr 1e-3"
)
BATCH_SIZE = 8
N_STEPS = 16
HW = 128
PARAMS = SimpleNamespace(
    patch_size=[16, 16],
    embed_dim=192,
    num_heads=3,
    processor_blocks=12,
    n_states=12,
    block_type="axial",
    time_type="attention",
    space_type="axial_attention",
    bias_type="rel",
    gradient_checkpointing=False,
)


def build_model():
    mpp = C.ppcfd_module("multiple_physics_pretraining")
    drop_path = C.ppcfd_module("multiple_physics_pretraining.DropPath_util")
    model = mpp.build_avit(PARAMS)
    for layer in model.sublayers():
        if isinstance(layer, drop_path.DropPath):
            layer.drop_prob = 0.0
    return model


def build_batch():
    inputs = {
        "x": C.randn([N_STEPS, BATCH_SIZE, 1, HW, HW]),
        "state_labels": paddle.zeros([BATCH_SIZE, 1], dtype="int64"),
        "bcs": paddle.zeros([BATCH_SIZE, 2], dtype="int64"),
    }
    return inputs, {"y": C.randn([BATCH_SIZE, 1, HW, HW])}


def forward(model, inputs):
    return model(inputs["x"], inputs["state_labels"], inputs["bcs"])


def compute_loss(model, inputs, labels):
    # train_basic.py: per-sample/channel NMSE, accum_grad = 1
    out, tar = forward(model, inputs), labels["y"]
    tar_norm = 1e-7 + tar.pow(2).mean(axis=(2, 3), keepdim=True)
    return ((out - tar).pow(2).mean(axis=(2, 3), keepdim=True) / tar_norm).mean()
