import os
from logging import getLogger

import paddle

from ...paddle_utils import *

from .embedders import LinearPointEmbedder
from .model_wrapper import ModelWrapper
from .sklearn_wrapper import SymbolicTransformerRegressor
from .transformer import TransformerModel

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching
    modules["encoder"] = TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    modules["decoder"] = TransformerModel(
        params,
        env.equation_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = paddle.load(path=str(params.reload_model))
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            # PaddlePaddle: 使用 .numel() 获取参数数量，兼容两个框架
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )
    if not params.cpu:
        from ...paddle_utils import device2str
        device_str = device2str(params.device)
        print(f"Using device: {device_str}")

        for v in modules.values():
            v.to(device_str)
    return modules
