"""UPT (RansSimformerNognnSdfModel) on ShapeNetCar pressure
(examples/UPT/yamls/shapenetcar/upt/dim768_seq1024sdf512_cnext_lr5e4_sd02_reprcnn_grn_grid32.yaml)."""
import importlib.util
import os
import sys

import paddle

from cases import _common as C


EXAMPLE = "UPT"
NOTE = (
    "synthetic batch of 16 cars x 3586 surface points in (0, 200) and a 32^3 SDF + 3 pos "
    "channels instead of the preprocessed ShapeNetCar .th files; query points = mesh "
    "points as with num_query_points_ratio=None; latent drop_path_rate 0.2 set to 0 so both "
    "modes are deterministic; trainer bf16 autocast not used (DotProductAttention still "
    "casts q/k/v to bf16 itself); harness Adam, the example AdamW wd 0.05 + warmup-cosine"
)
BATCH_SIZE = 16  # max_batch_size, effective 32 via 2 accumulation steps
N_POINTS = 3586
GRID = 32
LEARNING_RATE = 5e-4
# SgdTrainer.wrap_compile uses paddle.jit.to_static(model, backend="CINN") without
# full_graph, i.e. SOT: the dense-batching helpers call int(batch.max()) and loop over
# per-sample counts, which AST full_graph=True cannot trace
FULL_GRAPH = False
# With SOT dynamic shapes the einops.rearrange calls in kappamodules LayerNorm3d get a
# symbolic perm and Tensor.transpose fails ("'transpose' is not a callable object");
# every UPT step has the same shapes here, so trace with static shapes instead
os.environ.setdefault("SOT_ALLOW_DYNAMIC_SHAPE", "False")


def _setup_paths():
    """examples/UPT/main_train.py::_setup_paths_and_aliases"""
    upt_example = C.example_dir("UPT")
    source_upt = C.REPO_ROOT / "source" / "UPT"
    for path in (upt_example, C.REPO_ROOT, source_upt, source_upt / "KappaModules"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if "models" not in sys.modules:
        models_dir = C.REPO_ROOT / "ppcfd" / "models" / "UPT"
        spec = importlib.util.spec_from_file_location(
            "models", models_dir / "__init__.py", submodule_search_locations=[str(models_dir)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["models"] = module
        spec.loader.exec_module(module)
    return sys.modules["models"]


def build_model():
    models = _setup_paths()
    model = models.model_from_kwargs(
        kind="rans_simformer_nognn_sdf_model",
        input_shape=(None, 3),
        output_shape=(None, 1),
        grid_encoder=dict(
            kind="encoders.rans_grid_convnext",
            patch_size=2,
            kernel_size=3,
            depthwise=False,
            global_response_norm=True,
            depths=[2, 2, 2],
            dims=[192, 384, 768],
            upsample_size=64,
            upsample_mode="nearest",
            resolution=(GRID, GRID, GRID),
            concat_pos_to_sdf=True,
        ),
        mesh_encoder=dict(
            kind="encoders.rans_perceiver",
            num_output_tokens=1024,
            add_type_token=True,
            init_weights="truncnormal",
            kwargs=dict(dim=768, num_attn_heads=12),
        ),
        latent=dict(
            kind="latent.transformer_model",
            init_weights="truncnormal",
            drop_path_rate=0.0,
            drop_path_decay=False,
            kwargs=dict(dim=768, num_attn_heads=12, depth=12),
        ),
        decoder=dict(
            kind="decoders.rans_perceiver",
            init_weights="truncnormal",
            kwargs=dict(dim=768, num_attn_heads=12),
        ),
    )
    model.initialize_weights()
    # the trainer also calls train()/eval() on a wrapping TrainerModel: ModelBase.train
    # forwards mode= to paddle.nn.Layer.train, which Paddle 3.2 does not accept
    return Wrapper(model)


class Wrapper(paddle.nn.Layer):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        return self.model(**inputs)


def build_batch():
    mesh_pos = C.uniform([BATCH_SIZE * N_POINTS, 3], 1.0, 199.0)
    batch_idx = paddle.arange(BATCH_SIZE).repeat_interleave(N_POINTS)
    sdf = C.randn([BATCH_SIZE, GRID, GRID, GRID, 1])
    axis = paddle.linspace(-1.0, 1.0, GRID)
    pos = paddle.stack(paddle.meshgrid(axis, axis, axis), axis=-1)
    inputs = {
        "mesh_pos": mesh_pos,
        "sdf": paddle.concat([sdf, pos.unsqueeze(0).tile([BATCH_SIZE, 1, 1, 1, 1])], -1),
        "query_pos": mesh_pos.reshape([BATCH_SIZE, N_POINTS, 3]),
        "batch_idx": batch_idx,
        "unbatch_idx": batch_idx.clone(),  # no padding: every car has N_POINTS queries
        "unbatch_select": paddle.arange(BATCH_SIZE),
    }
    return inputs, {"pressure": C.randn([BATCH_SIZE * N_POINTS, 1])}


def forward(model, inputs):
    return model(**inputs)["x_hat"]


def compute_loss(model, inputs, labels):
    # elementwise_loss(mse_loss) with reduction="mean"
    return C.mse(forward(model, inputs), labels["pressure"])
