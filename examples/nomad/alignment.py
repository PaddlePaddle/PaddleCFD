import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import paddle

from ppcfd.models.nomad.advection.operator_model import OperatorModel as AdvectionModel
from ppcfd.models.nomad.antiderivative.nomad_antiderivative import (
    OperatorModel as AntiderivativeModel,
)
from ppcfd.models.nomad.shallowwater.nomad_model import OperatorModel as ShallowWaterModel


CASE_CONFIGS = {
    "antiderivative": {
        "model": AntiderivativeModel,
        "m": 500,
        "p": 32,
        "du": 1,
        "dy": 1,
        "ds": 1,
        "batch": 3,
    },
    "advection": {
        "model": AdvectionModel,
        "m": 256,
        "p": 64,
        "du": 1,
        "dy": 2,
        "ds": 1,
        "batch": 2,
    },
    "shallowwater": {
        "model": ShallowWaterModel,
        "m": 1024,
        "p": 32,
        "du": 3,
        "dy": 3,
        "ds": 3,
        "batch": 2,
    },
}


def make_layers(cfg, n, decoder):
    branch_in = cfg["m"] * cfg["du"]
    if decoder == "nonlinear":
        return (
            [branch_in, 100, 100, 100, 100, 100, cfg["ds"] * n],
            [cfg["ds"] * n * 2, 100, 100, 100, 100, 100, cfg["ds"]],
        )
    if decoder == "linear":
        return (
            [branch_in, 100, 100, 100, 100, 100, cfg["ds"] * n],
            [cfg["dy"], 100, 100, 100, 100, 100, cfg["ds"] * n],
        )
    raise ValueError("decoder must be 'linear' or 'nonlinear'")


def init_weights(layers, rng):
    params = []
    for in_dim, out_dim in zip(layers[:-1], layers[1:]):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        weight = rng.normal(0.0, scale, size=(in_dim, out_dim)).astype("float32")
        bias = rng.normal(0.0, scale, size=(out_dim,)).astype("float32")
        params.append((weight, bias))
    return params


def assign_mlp(layer, params):
    linear_idx = 0
    state = layer.state_dict()
    new_state = {}
    for name, tensor in state.items():
        weight, bias = params[linear_idx]
        if name.endswith(".weight"):
            new_state[name] = paddle.to_tensor(weight)
        elif name.endswith(".bias"):
            new_state[name] = paddle.to_tensor(bias)
            linear_idx += 1
        else:
            new_state[name] = tensor
    layer.set_state_dict(new_state)


def jax_mlp(params, x):
    for i, (weight, bias) in enumerate(params):
        x = jnp.matmul(x, weight) + bias
        if i != len(params) - 1:
            x = jax.nn.gelu(x, approximate=False)
    return x


def jax_forward(params, inputs, cfg, n, decoder):
    trunk_params, branch_params = params
    u, y = inputs
    batch = u.shape[0]

    if decoder == "nonlinear":
        b = jax_mlp(branch_params, u.reshape(batch, 1, cfg["m"] * cfg["du"]))
        b = jnp.tile(b, (1, y.shape[1], 1))
        y_tiled = jnp.tile(y, (1, 1, b.shape[-1] // y.shape[-1]))
        return jax_mlp(trunk_params, jnp.concatenate((y_tiled, b), axis=-1))

    t = jax_mlp(trunk_params, y)
    t = t.reshape(y.shape[0], y.shape[1], cfg["ds"], n)
    b = jax_mlp(branch_params, u.reshape(batch, 1, cfg["m"] * cfg["du"]))
    b = b.reshape(b.shape[0], b.shape[2] // cfg["ds"], cfg["ds"])
    return jnp.einsum("ijkl,ilk->ijk", t, b)


def jax_loss(params, inputs, target, cfg, n, decoder):
    pred = jax_forward(params, inputs, cfg, n, decoder)
    return jnp.mean((target.reshape(-1) - pred.reshape(-1)) ** 2)


def jax_sgd_step(params, inputs, target, cfg, n, decoder, lr):
    grads = jax.grad(jax_loss)(params, inputs, target, cfg, n, decoder)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def paddle_loss(model, u, y, target):
    return model.loss(((u, y), target))


def paddle_sgd_step(model, optimizer, u, y, target):
    loss = paddle_loss(model, u, y, target)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    return float(loss.numpy())


def max_tree_diff(jax_tree, paddle_params):
    jax_arrays = [
        np.asarray(x)
        for x in jax.tree_util.tree_leaves(jax_tree)
        if hasattr(x, "shape")
    ]
    diffs = []
    for jax_arr, paddle_arr in zip(jax_arrays, paddle_params):
        diffs.append(float(np.max(np.abs(jax_arr - paddle_arr))))
    return max(diffs) if diffs else 0.0


def run_case(case, decoder, n, steps, lr, seed):
    cfg = CASE_CONFIGS[case]
    branch_layers, trunk_layers = make_layers(cfg, n, decoder)
    rng = np.random.default_rng(seed)

    branch_params = init_weights(branch_layers, rng)
    trunk_params = init_weights(trunk_layers, rng)
    jax_params = (trunk_params, branch_params)

    model_cls = cfg["model"]
    if case == "shallowwater":
        model = model_cls(branch_layers, trunk_layers, n=n, decoder=decoder, ds=cfg["ds"])
    else:
        model = model_cls(
            branch_layers,
            trunk_layers,
            m=cfg["m"],
            P=cfg["p"],
            n=n,
            decoder=decoder,
            ds=cfg["ds"],
        )

    assign_mlp(model.branch_net, branch_params)
    assign_mlp(model.trunk_net, trunk_params)

    u = rng.normal(size=(cfg["batch"], cfg["m"], cfg["du"])).astype("float32")
    y = rng.normal(size=(cfg["batch"], cfg["p"], cfg["dy"])).astype("float32")
    target = rng.normal(size=(cfg["batch"], cfg["p"], cfg["ds"])).astype("float32")

    jax_inputs = (jnp.asarray(u), jnp.asarray(y))
    jax_target = jnp.asarray(target)
    paddle_u = paddle.to_tensor(u)
    paddle_y = paddle.to_tensor(y)
    paddle_target = paddle.to_tensor(target)

    jax_pred = np.asarray(jax_forward(jax_params, jax_inputs, cfg, n, decoder))
    paddle_pred = model.predict((paddle_u, paddle_y)).numpy()
    jax_forward_loss = float(jax_loss(jax_params, jax_inputs, jax_target, cfg, n, decoder))
    paddle_forward_loss = float(paddle_loss(model, paddle_u, paddle_y, paddle_target).numpy())

    jax_grads = jax.grad(jax_loss)(jax_params, jax_inputs, jax_target, cfg, n, decoder)
    paddle_train_loss = paddle_loss(model, paddle_u, paddle_y, paddle_target)
    paddle_train_loss.backward()
    paddle_grad_arrays = [param.grad.numpy() for param in model.trunk_net.parameters()]
    paddle_grad_arrays += [param.grad.numpy() for param in model.branch_net.parameters()]
    grad_max_abs_diff = max_tree_diff(jax_grads, paddle_grad_arrays)
    model.clear_gradients()

    optimizer = paddle.optimizer.SGD(learning_rate=lr, parameters=model.parameters())
    jax_trace = []
    paddle_trace = []
    for _ in range(steps):
        jax_trace.append(float(jax_loss(jax_params, jax_inputs, jax_target, cfg, n, decoder)))
        paddle_trace.append(paddle_sgd_step(model, optimizer, paddle_u, paddle_y, paddle_target))
        jax_params = jax_sgd_step(jax_params, jax_inputs, jax_target, cfg, n, decoder, lr)

    jax_trace.append(float(jax_loss(jax_params, jax_inputs, jax_target, cfg, n, decoder)))
    paddle_trace.append(float(paddle_loss(model, paddle_u, paddle_y, paddle_target).numpy()))

    return {
        "case": case,
        "decoder": decoder,
        "n": n,
        "forward_loss_jax": jax_forward_loss,
        "forward_loss_paddle": paddle_forward_loss,
        "forward_loss_abs_diff": abs(jax_forward_loss - paddle_forward_loss),
        "prediction_max_abs_diff": float(np.max(np.abs(jax_pred - paddle_pred))),
        "grad_max_abs_diff": grad_max_abs_diff,
        "train_loss_trace_jax": jax_trace,
        "train_loss_trace_paddle": paddle_trace,
        "train_loss_trace_max_abs_diff": float(
            np.max(np.abs(np.asarray(jax_trace) - np.asarray(paddle_trace)))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="NOMAD JAX/Paddle alignment check")
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS) + ["all"], default="all")
    parser.add_argument("--decoder", choices=["linear", "nonlinear"], default="nonlinear")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("alignment_report.json"))
    args = parser.parse_args()

    paddle.set_device("cpu")
    cases = sorted(CASE_CONFIGS) if args.case == "all" else [args.case]
    results = [run_case(case, args.decoder, args.n, args.steps, args.lr, args.seed) for case in cases]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
