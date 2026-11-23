"""CoNFILD conditional neural field: SIREN-FiLM decoder + latent codes (examples/confild/conf/confild_main.yaml)."""
import paddle

from cases import _common as C


EXAMPLE = "confild"
NOTE = (
    "synthetic [64, 1024, 3] fields on 1024 random coords in [-1, 1] (the size written by "
    "generate_test_data.py); one Adam at the field lr 1e-4 instead of 1e-4 field + 1e-5 latents"
)
BATCH_SIZE = 64
N_POINTS = 1024
LEARNING_RATE = 1e-4


class CoNFILD(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        confild = C.ppcfd_module("confild")
        self.confild = confild.SIRENAutodecoder_film(
            input_keys=["confild_x", "latent_z"],
            output_keys=["confild_output"],
            num_hidden_layers=10,
            out_features=3,
            hidden_features=128,
            in_coord_features=2,
            in_latent_features=128,
        )
        self.latent = confild.LatentContainer(
            input_keys=["latent_x"],
            output_keys=["latent_z"],
            N_samples=16000,
            lumped=True,
            N_features=128,
            dims=2,
        )

    def forward(self, inputs):
        latent_z = self.latent({"latent_x": inputs["latent_x"]})["latent_z"]
        coords = inputs["coords"].unsqueeze(0).tile([inputs["latent_x"].shape[0], 1, 1])
        return self.confild({"confild_x": coords, "latent_z": latent_z})["confild_output"]


def build_model():
    return CoNFILD()


def build_batch():
    inputs = {
        "latent_x": paddle.randperm(16000)[:BATCH_SIZE],
        "coords": C.uniform([N_POINTS, 2], -1.0, 1.0),
    }
    return inputs, {"fields": C.uniform([BATCH_SIZE, N_POINTS, 3], -1.0, 1.0)}


def compute_loss(model, inputs, labels):
    return C.mse(model(inputs), labels["fields"])
