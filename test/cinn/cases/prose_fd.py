"""PROSE-FD (PROSE_2to1, data + symbol fusion) on 2D fluids (examples/prose_fd/configs/main.yaml)."""
import paddle
from omegaconf import OmegaConf

from cases import _common as C


EXAMPLE = "prose_fd"
NOTE = (
    "synthetic meanvar-normalized [8, 10, 128, 128, 4] input / next-10-step label and "
    "random equation token ids (last 10 of 50 padded) instead of PDEBench/PDEArena; "
    "batch 8 instead of 128; fp32 instead of bf16 autocast; harness uses Adam without "
    "grad clipping, the example AdamW wd 1e-4 + clip 1"
)
BATCH_SIZE = 8
X_NUM = 128
T_NUM = 20
INPUT_LEN = 10
MAX_OUTPUT_DIM = 4
SYMBOL_LEN = 50
N_PAD = 10
LEARNING_RATE = 1e-4
CONFIGS = C.example_dir("prose_fd") / "configs"
SYMBOL_ENV = None


def build_model():
    global SYMBOL_ENV
    prose = C.ppcfd_module("prose_fd")
    SYMBOL_ENV = prose.SymbolicEnvironment(OmegaConf.load(CONFIGS / "symbol" / "symbol.yaml"))
    model_cfg = OmegaConf.load(CONFIGS / "model" / "prose_2to1.yaml")
    return prose.PROSE_2to1(
        model_cfg, SYMBOL_ENV, X_NUM, MAX_OUTPUT_DIM, output_len=T_NUM - INPUT_LEN
    )


def build_batch():
    n_words = len(SYMBOL_ENV.equation_id2word)
    symbols = C.randint([BATCH_SIZE, SYMBOL_LEN], n_words)
    padding = paddle.arange(SYMBOL_LEN).unsqueeze(0).tile([BATCH_SIZE, 1]) >= SYMBOL_LEN - N_PAD
    symbols = paddle.where(padding, paddle.full_like(symbols, SYMBOL_ENV.pad_index), symbols)
    t = paddle.linspace(0, 10, T_NUM, dtype="float32")[None]  # Trainer.t, use_raw_time=0
    inputs = {
        "data_input": C.randn([BATCH_SIZE, INPUT_LEN, X_NUM, X_NUM, MAX_OUTPUT_DIM]),
        "input_times": t[:, :INPUT_LEN, None],
        "output_times": t[:, INPUT_LEN:, None] - t[:, INPUT_LEN - 1 : INPUT_LEN, None],
        "symbol_input": symbols,
        "symbol_padding_mask": padding,
    }
    labels = {
        "data_label": C.randn([BATCH_SIZE, T_NUM - INPUT_LEN, X_NUM, X_NUM, MAX_OUTPUT_DIM]),
        "data_mask": paddle.ones([BATCH_SIZE, 1, 1, 1, MAX_OUTPUT_DIM], dtype="float32"),
    }
    return inputs, labels


def forward(model, inputs):
    return model(
        "fwd",
        inputs["data_input"],
        inputs["input_times"],
        inputs["output_times"],
        inputs["symbol_input"],
        inputs["symbol_padding_mask"],
    )


def compute_loss(model, inputs, labels):
    # Trainer.iter + data_loss_fn with loss_weight=none, denormalize_for_loss=0
    out = forward(model, inputs) * labels["data_mask"]
    loss = paddle.nn.functional.mse_loss(out, labels["data_label"], reduction="none")
    return loss.sum() / paddle.count_nonzero(labels["data_mask"].expand_as(loss))
