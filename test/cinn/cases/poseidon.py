"""Poseidon ScOT-T on the compressible Airfoil (examples/poseidon/configs/run_small.yaml, model "T")."""
from cases import _common as C


EXAMPLE = "poseidon"
NOTE = (
    "synthetic [32, 1, 128, 128] fields instead of the SE-AF .nc data; batch 32 as in "
    "examples/poseidon/benchmark_compiler.py (run_small.yaml trains with 2); airfoil "
    "pixel_mask omitted; needs Paddle >= 3.3 (LayerNorm(eps=), paddle.compat.nn); harness "
    "uses Adam without grad clipping, the example AdamW wd 1e-6 + clip 5 + cosine"
)
BATCH_SIZE = 32
RES = 128
LEARNING_RATE = 5e-5


def build_model():
    poseidon = C.ppcfd_module("poseidon")
    config = poseidon.ScOTConfig(
        image_size=RES,
        patch_size=4,
        num_channels=1,
        num_out_channels=1,
        embed_dim=48,
        depths=[4, 4, 4, 4],
        num_heads=[3, 6, 12, 24],
        skip_connections=[2, 2, 2, 0],
        window_size=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=[0, 1],
        residual_model="convnext",
        use_conditioning=False,
        learn_residual=False,
    )
    return poseidon.ScOT(config)


def build_batch():
    return {"x": C.randn([BATCH_SIZE, 1, RES, RES])}, {"y": C.randn([BATCH_SIZE, 1, RES, RES])}


def forward(model, inputs):
    return model(pixel_values=inputs["x"], return_dict=False)[0]


def compute_loss(model, inputs, labels):
    # ScOT computes the channel-normalized L1 loss itself when labels are given
    return model(pixel_values=inputs["x"], labels=labels["y"], return_dict=False)[0]
