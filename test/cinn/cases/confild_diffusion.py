"""CoNFILD latent diffusion UNet, eps-prediction MSE (examples/confild/conf/un_confild_case1.yaml)."""
import sys

import paddle

from cases import _common as C


EXAMPLE = "confild"
NOTE = (
    "synthetic latent images [16, 1, 128, 128] in [-1, 1]; timesteps and noise are drawn "
    "once into the batch so both modes see the same diffusion sample; harness uses Adam, "
    "the example AdamW (weight_decay 0)"
)
BATCH_SIZE = 16
IMAGE_SIZE = 128
STEPS = 1000
LEARNING_RATE = 5e-5


def _schedule():
    sys.path.insert(0, str(C.example_dir("confild")))
    from un_confild_main import get_named_beta_schedule

    return get_named_beta_schedule("cosine", STEPS)


class LatentDiffusion(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        confild = C.ppcfd_module("confild")
        self.unet = confild.UNetModel(
            image_size=IMAGE_SIZE,
            in_channels=1,
            model_channels=128,
            out_channels=1,
            num_res_blocks=2,
            attention_resolutions=(32, 16, 8),
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
            dims=2,
        )
        self.diffusion = confild.SpacedDiffusion(
            use_timesteps=range(STEPS),
            betas=_schedule(),
            model_mean_type=confild.ModelMeanType.EPSILON,
            model_var_type=confild.ModelVarType.FIXED_LARGE,
            loss_type=confild.LossType.MSE,
            rescale_timesteps=False,
        )

    def forward(self, inputs):
        return self.unet(inputs["x_t"], inputs["t"])

    def loss(self, inputs):
        terms = self.diffusion.training_losses(
            self.unet, inputs["x_start"], inputs["t"], noise=inputs["noise"]
        )
        return terms["loss"].mean()  # UniformSampler weights are all 1


def build_model():
    return LatentDiffusion()


def build_batch():
    shape = [BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE]
    inputs = {
        "x_start": C.uniform(shape, -1.0, 1.0),
        "x_t": C.randn(shape),
        "t": C.randint([BATCH_SIZE], STEPS),
        "noise": C.randn(shape),
    }
    return inputs, {}


def forward(model, inputs):
    return model(inputs)


def compute_loss(model, inputs, labels):
    return model.loss(inputs)
