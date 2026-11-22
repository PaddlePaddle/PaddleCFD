"""TransKAN (Transolver + KAN head) on the ventilation cavity (examples/ventilation_cavity/config.yaml)."""
from cases import _common as C


EXAMPLE = "ventilation_cavity"
NOTE = (
    "synthetic [256, 100, 8] input / [256, 50, 2] target instead of data_test.csv; "
    "dropout 0.15 disabled by the harness; kan_grid_range stays at the default [-1, 1] "
    "because the config key is misspelled kan_gird_range; harness uses Adam, the example AdamW"
)
BATCH_SIZE = 256
LEARNING_RATE = 1e-4


def build_model():
    transkan = C.ppcfd_module("ppTransKAN.transkan")
    return transkan.Transolver(
        space_dim=8,
        n_layers=3,
        n_hidden=128,
        dropout=0.15,
        n_head=8,
        act="gelu",
        mlp_ratio=1,
        fun_dim=0,
        out_dim=2,
        slice_num=16,
        apply_kan=True,
        kan_hidden_layer=[100, 32, 50],
        kan_grid_size=16,
        kan_gird_range=[-2, 0.5],
        t_len_out=50,
    )


def build_batch():
    return {"x": C.randn([BATCH_SIZE, 100, 8])}, {"y": C.randn([BATCH_SIZE, 50, 2])}


def forward(model, inputs):
    return model(inputs["x"])


def compute_loss(model, inputs, labels):
    return C.l1(forward(model, inputs), labels["y"])
