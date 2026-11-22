"""Neural-operator Unet2D for the airfoil wake, NO stage (examples/airfoil_wake/nodm/train_no.py)."""
import paddle

from cases import _common as C


EXAMPLE = "airfoil_wake/nodm"
NOTE = (
    "synthetic UX history [20, 10, 1, 128, 512] / next frame instead of UX_nan_filtered.npy; "
    "shift 0 / scale 1 normalization and a fixed rectangular airfoil mask instead of mask.npy; "
    "time drawn from time_cond = linspace(0, 1, lf=2); the example trains with AMP GradScaler, "
    "the harness in fp32"
)
BATCH_SIZE = 20
NX, NY = 128, 512
LB, LF = 10, 2
LEARNING_RATE = 5e-5
PAR = {"nx": NX, "ny": NY, "nf": 1, "lb": LB, "lf": LF, "LF": 10, "channels": LB}
PAR.update(inp_shift=0.0, inp_scale=1.0, out_shift=0.0, out_scale=1.0, t_shift=0.0, t_scale=1.0)


def build_model():
    matcho = C.ppcfd_module("ppdiffusion.matcho", optional_deps=("xskillscore",))
    mask = paddle.ones([1, 1, NX, NY], dtype="float32")
    mask[:, :, 48:80, 96:160] = 0.0  # airfoil body
    par = dict(PAR, mask=mask)
    return matcho.Unet2D(dim=16, Par=par, dim_mults=(1, 2, 4, 8), channels=par["channels"])


def build_batch():
    inputs = {
        "x": C.randn([BATCH_SIZE, LB, 1, NX, NY]),
        "time": C.randint([BATCH_SIZE], LF).astype("float32") / (LF - 1),
    }
    return inputs, {"y": C.randn([BATCH_SIZE, 1, NX, NY])}


def forward(model, inputs):
    return model(inputs["x"], inputs["time"])


def compute_loss(model, inputs, labels):
    # CustomLoss: global relative L2 on the normalized fields
    y_true = (labels["y"] - PAR["out_shift"]) / PAR["out_scale"]
    y_pred = (forward(model, inputs) - PAR["out_shift"]) / PAR["out_scale"]
    return paddle.linalg.norm(y_true - y_pred, p=2) / paddle.linalg.norm(y_true, p=2)
