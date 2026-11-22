"""PhysicsRegression (PhyE2E) embedder + encoder + decoder, teacher-forced CE
(examples/physics_regression/bash/train_small_cinn.sh)."""
import sys

import paddle

from cases import _common as C


EXAMPLE = "physics_regression"
NOTE = (
    "random float-token ids [200 points, 32 eqs, 37] and equation-token ids [40, 32] in "
    "place of generated expressions; the numpy float->token encoding and hint encoding "
    "of LinearPointEmbedder.forward run outside the model, the harness starts at "
    "embed + compress (compress re-written with reshape instead of Tensor.view); every "
    "sequence is full length (no padding); 32 equations per "
    "step (tokens_per_batch 5000 / 200 points ~ 25) instead of batch_size 256; harness "
    "Adam lr 1e-4 instead of adam_inverse_sqrt warmup"
)
BATCH_SIZE = 32
N_POINTS = 200
EQ_LEN = 40
LEARNING_RATE = 1e-4
ENV = None


def _params():
    example = C.example_dir("physics_regression")
    sys.path.insert(0, str(example))
    from parsers import get_parser

    return get_parser().parse_args(
        [
            "--max_len", str(N_POINTS),
            "--max_input_points", str(N_POINTS),
            "--tokens_per_batch", "5000",
            "--add_consts", "1",
            "--device", "cuda:0",
            "--use_dimension_mask", "0",
            "--decode_physical_units", "single-seq",
            "--use_hints", "units,complexity,unarys,consts",
            "--random_variables_sequence", "0",
            "--rescale", "0",
            "--feynman_equations_path", str(example / "data" / "FeynmanEquations.xlsx"),
            "--units_csv_path", str(example / "data" / "units.csv"),
            "--expr_test_data_path", "",  # default exprs_test_ranked.json is not shipped
        ]
    )


class PhyE2E(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        global ENV
        envs = C.ppcfd_module("physicsregression.symbolicregression.envs")
        model = C.ppcfd_module("physicsregression.symbolicregression.model")
        params = _params()
        ENV = envs.build_env(params)
        modules = model.build_modules(ENV, params)
        self.embedder = modules["embedder"]
        self.encoder = modules["encoder"]
        self.decoder = modules["decoder"]

    def compress(self, emb):
        # LinearPointEmbedder.compress with reshape instead of Tensor.view, which
        # to_static rejects; the example's CINN path (PHYE2E_USE_CINN) only compiles
        # the encoder and decoder, so the embedder never meets to_static there
        max_len, bs = emb.shape[0], emb.shape[1]
        x = emb.reshape([max_len, bs, -1])
        for layer in self.embedder.hidden_layers:
            x = self.embedder.activation_fn(layer(x))
        return self.embedder.fc(x)

    def forward(self, x1, len1, x2, len2):
        x1 = self.compress(self.embedder.embed(x1))
        encoded = self.encoder("fwd", x=x1, lengths=len1, causal=False)
        return self.decoder(
            "fwd",
            x=x2,
            lengths=len2,
            causal=True,
            src_enc=encoded.transpose([1, 0, 2]),
            src_len=len1,
            units=None,
        )


def build_model():
    return PhyE2E()


def build_batch():
    # LinearPointEmbedder.batch: one point = float_vector_descriptor_len + 2 tokens
    desc = 2 + (2 + ENV.params.mantissa_len) * (
        ENV.params.max_input_dimension + ENV.params.max_output_dimension
    )
    inputs = {
        "x1": C.randint([N_POINTS, BATCH_SIZE, desc], len(ENV.float_id2word)),
        "len1": paddle.full([BATCH_SIZE], N_POINTS, dtype="int64"),
        "x2": C.randint([EQ_LEN, BATCH_SIZE], len(ENV.equation_id2word)),
        "len2": paddle.full([BATCH_SIZE], EQ_LEN, dtype="int64"),
    }
    return inputs, {}


def forward(model, inputs):
    return model(inputs["x1"], inputs["len1"], inputs["x2"], inputs["len2"])


def compute_loss(model, inputs, labels):
    # Trainer.enc_dec_step: predict every next equation token
    x2, len2 = inputs["x2"], inputs["len2"]
    alen = paddle.arange(EQ_LEN, dtype="int64")
    pred_mask = alen[:, None] < len2[None] - 1
    y = x2[1:].masked_select(pred_mask[:-1])
    decoded = forward(model, inputs)
    _, loss = model.decoder(
        "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False, y_units=None
    )
    return loss
