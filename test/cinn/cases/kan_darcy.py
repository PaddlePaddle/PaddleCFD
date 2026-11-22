"""RBF KAN DeepONet (KANONet) on Darcy flow (examples/darcyflow/ppkan/conf/main.yaml)."""
from cases import _common as C


EXAMPLE = "darcyflow/ppkan"
NOTE = (
    "synthetic coefficient [32, 841] and solution in place of piececonst_r421 (subsampled "
    "29x29); trunk input is the shared 841-point grid; harness Adam has no weight_decay=1e-4"
)
BATCH_SIZE = 32
N_GRID = 29 * 29


def build_model():
    kan_rbf = C.ppcfd_module("ppkan.kan_rbf")
    return kan_rbf.KANONet(
        width_trunk=[2, 64, 40],
        width_branch=[N_GRID, 64, 40],
        trunk_min_grid=0.0,
        trunk_max_grid=1.0,
        branch_min_grid=0.0,
        branch_max_grid=1.0,
        grid_count=3,
        grid_opt=False,
        branch_kan_func="rbf",
        trunk_kan_func="rbf",
        apply_base_update=True,
        noise_scale=0.01,
        dtype="float32",
    )


def build_batch():
    inputs = {"c": C.uniform([BATCH_SIZE, N_GRID]), "x": C.uniform([N_GRID, 2])}
    return inputs, {"y": C.randn([BATCH_SIZE, N_GRID])}


def forward(model, inputs):
    return model(inputs["c"], inputs["x"])


def compute_loss(model, inputs, labels):
    return C.mse(forward(model, inputs), labels["y"])
