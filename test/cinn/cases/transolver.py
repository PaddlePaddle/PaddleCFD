"""Transolver on ShapeNetCar (examples/aerodynamic_car_design/main_shapenetcar.py)."""
import paddle

from cases import _common as C


EXAMPLE = "aerodynamic_car_design"
NOTE = (
    "synthetic [32186, 7] pos+sdf+normal input and [1, 32186, 4] targets, first 3586 "
    "points flagged as surface (ShapeNetCar sizes); kwargs hard-coded in "
    "main_shapenetcar.py, the yaml model block is unused there"
)
BATCH_SIZE = 1
N_POINTS = 32186
N_SURF = 3586


def build_model():
    transolver = C.ppcfd_module("pptransformer.Transolver")
    return transolver.Transolver(
        n_hidden=256,
        n_layers=8,
        space_dim=7,
        fun_dim=0,
        n_head=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=32,
        unified_pos=False,
    )


def build_batch():
    surf = paddle.arange(N_POINTS) < N_SURF
    labels = {"targets": C.randn([BATCH_SIZE, N_POINTS, 4]), "surf": surf.unsqueeze(0)}
    return {"x": C.randn([N_POINTS, 7])}, labels


def forward(model, inputs):
    return model(inputs["x"])


def compute_loss(model, inputs, labels):
    out = model(inputs["x"])
    targets, surf = labels["targets"], labels["surf"]
    p_pred = paddle.stack([out[i][surf[i], -1:] for i in range(BATCH_SIZE)], 0)
    p_true = paddle.stack([targets[i][surf[i], -1:] for i in range(BATCH_SIZE)], 0)
    loss_press = C.mse(p_pred, p_true)
    loss_velo = C.mse(out[:, :, :-1], targets[:, :, :-1])
    return loss_velo + loss_press
