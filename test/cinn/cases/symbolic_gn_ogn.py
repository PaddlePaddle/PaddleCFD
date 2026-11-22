"""Symbolic GN: OGN message-passing net on the 6-body spring system (examples/symbolic_gn/conf/config.yaml)."""
from cases import _common as C


EXAMPLE = "symbolic_gn"
NOTE = (
    "synthetic node states [32, 6, 6] (pos, vel, charge, mass>0) and accelerations; "
    "edge_index is handed to the model as the numpy array get_edge_index returns, because "
    "OGN.forward calls .numpy() on a Tensor edge_index, which to_static cannot trace; "
    "harness Adam has no weight_decay=1e-8 / cosine schedule"
)
BATCH_SIZE = 32
N_NODES = 6
N_F = 6
NDIM = 2
EDGE_INDEX = None


def build_model():
    global EDGE_INDEX
    gn = C.ppcfd_module("symbolic_gn")
    EDGE_INDEX = gn.get_edge_index(N_NODES, "spring")
    return gn.OGN(
        n_f=N_F,
        msg_dim=100,
        ndim=NDIM,
        hidden=300,
        edge_index=EDGE_INDEX,
        l1_strength=1e-2,
    )


def build_batch():
    x = C.randn([BATCH_SIZE, N_NODES, N_F])
    x[:, :, -1] = 1.0 + C.uniform([BATCH_SIZE, N_NODES])  # mass
    return {"x": x}, {"acceleration": C.randn([BATCH_SIZE, N_NODES, NDIM])}


def forward(model, inputs):
    return model({"x": inputs["x"], "edge_index": EDGE_INDEX})


def compute_loss(model, inputs, labels):
    out = forward(model, inputs)
    return C.l1(out["acceleration"], labels["acceleration"]) + out["l1_regularization"]
