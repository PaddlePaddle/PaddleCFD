"""G-FNO (GFNO2d, p4) on 2D reaction-diffusion, teacher forcing (examples/G-FNO/README.md rdb command)."""
import paddle

from cases import _common as C


EXAMPLE = "G-FNO"
NOTE = (
    "synthetic [32, 16, 16, 1, 1] previous step / next step instead of 2D_rdb_NA_NA.h5 "
    "(rdb_super_res 64 / rdb_downsample 4 -> S=16); uses paddle.fft and torch-compat APIs, "
    "needs Paddle >= 3.3; harness Adam has no weight_decay (lmbda) / cosine schedule"
)
BATCH_SIZE = 32
S = 16
# experiments.py (GFNO_USE_CINN=1) compiles with SOT: get_weight re-assigns self.weights
# inside forward, which AST full_graph=True rejects
FULL_GRAPH = False


def build_model():
    g_fno = C.ppcfd_module("g_fno")
    return g_fno.GFNO2d(
        num_channels=1, modes=8, width=10, initial_step=1, reflection=False, grid_type="symmetric"
    )


def build_batch():
    return {"x": C.randn([BATCH_SIZE, S, S, 1, 1])}, {"y": C.randn([BATCH_SIZE, S, S, 1, 1])}


def forward(model, inputs):
    return model(inputs["x"])


def compute_loss(model, inputs, labels):
    # LpLoss(size_average=False).rel on [B, -1, C]: per-sample relative L2, summed over batch
    im = forward(model, inputs).reshape([BATCH_SIZE, -1, 1])
    y = labels["y"].reshape([BATCH_SIZE, -1, 1])
    rel = paddle.norm(im - y, 2, 1) / paddle.norm(y, 2, 1)
    return paddle.sum(rel.mean(-1))
