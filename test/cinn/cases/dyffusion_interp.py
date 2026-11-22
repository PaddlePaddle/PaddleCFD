"""DYffusion interpolator SimpleUnet on Navier-Stokes flow past obstacles
(examples/flow_field_prediction/ppdiffusion/configs/config.yaml, process: interpolation)."""
from cases import _common as C


EXAMPLE = "flow_field_prediction/ppdiffusion"
NOTE = (
    "synthetic window+last-frame inputs [32, 6, 221, 42], condition [32, 2, 221, 42] and "
    "targets; the interpolation time (randint in [1, 15] in Interpolation.data_transform) "
    "is drawn once into the batch; dropout 0.15 disabled by the harness; harness uses "
    "Adam without grad clipping / accumulation, the example AdamW + ClipGradByNorm(1)"
)
BATCH_SIZE = 32
H, W = 221, 42
HORIZON = 16
LEARNING_RATE = 3e-4


def build_model():
    ppdiffusion = C.ppcfd_module("ppdiffusion", optional_deps=("xskillscore",))
    return ppdiffusion.SimpleUnet(
        dim=64,
        num_input_channels=3 * 2,  # Interpolation.model_cfg_transform, window=1
        num_output_channels=3,
        num_cond_channels=2,
        with_time_emb=True,
        outer_sample_mode="bilinear",
        upsample_dims=[256, 256],
        dropout=0.15,
        input_dropout=0.0,
        spatial_channels=[H, W],
    )


def build_batch():
    inputs = {
        "inputs": C.randn([BATCH_SIZE, 6, H, W]),
        "time": C.randint([BATCH_SIZE], HORIZON, low=1),
        "condition": C.randn([BATCH_SIZE, 2, H, W]),
    }
    return inputs, {"targets": C.randn([BATCH_SIZE, 3, H, W])}


def forward(model, inputs):
    return model(**inputs)


def compute_loss(model, inputs, labels):
    return C.mse(model(**inputs), labels["targets"])
