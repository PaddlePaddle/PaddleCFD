"""Benchmark the CINN compiler against the pure dynamic-graph path on ScOT.

The CINN FLAGS must be set before `import paddle`, so the two modes cannot live in
one process. This script therefore acts as its own driver: by default it spawns one
worker subprocess per mode (`dygraph`, `cinn`), each running the same synthetic
train step loop, then prints a comparison table.

Usage:
    python benchmark_compiler.py                      # both modes, ScOT-T, batch 32
    python benchmark_compiler.py --model B --steps 60
    python benchmark_compiler.py --modes cinn         # single mode only
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

MODE_ENV = "POSEIDON_USE_CINN"

# Same three FLAGS train.py sets, kept in sync deliberately: this benchmark must
# measure the configuration that training actually uses.
if os.environ.get(MODE_ENV, "0") == "1":
    os.environ["FLAGS_prim_enable_dynamic"] = "true"
    os.environ["FLAGS_prim_all"] = "true"
    os.environ["FLAGS_use_cinn"] = "true"
else:
    os.environ["FLAGS_prim_enable_dynamic"] = "false"
    os.environ["FLAGS_prim_all"] = "false"
    os.environ["FLAGS_use_cinn"] = "false"


# Mirrors MODEL_MAP in train.py. Duplicated on purpose: importing train.py would drag
# in wandb / matplotlib, which this benchmark does not need.
MODEL_MAP = {
    "T": {"depths": [4, 4, 4, 4], "embed_dim": 48},
    "S": {"depths": [8, 8, 8, 8], "embed_dim": 48},
    "B": {"depths": [8, 8, 8, 8], "embed_dim": 96},
    "L": {"depths": [8, 8, 8, 8], "embed_dim": 192},
}
COMMON_SCALE_ARGS = {
    "num_heads": [3, 6, 12, 24],
    "skip_connections": [2, 2, 2, 0],
    "window_size": 16,
    "patch_size": 4,
    "mlp_ratio": 4.0,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Compare CINN vs dynamic graph on ScOT.")
    parser.add_argument("--model", default="T", choices=["T", "S", "B", "L"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_channels", type=int, default=1, help="Input channels (SE-AF: 1).")
    parser.add_argument("--num_out_channels", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40, help="Timed steps after warm-up.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warm-up steps.")
    parser.add_argument(
        "--time_conditioning",
        action="store_true",
        help="Enable ConditionalLayerNorm time conditioning (time-dependent datasets).",
    )
    parser.add_argument(
        "--modes",
        default="dygraph,cinn",
        help="Comma-separated subset of 'dygraph,cinn'.",
    )
    parser.add_argument("--worker_mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result_path", default=None, help=argparse.SUPPRESS)
    return parser


# --------------------------------------------------------------------------- worker


def run_worker(args):
    import numpy as np
    import paddle

    # examples/poseidon/../.. is the PaddleCFD root that holds the ppcfd package.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from ppcfd.models.poseidon import ScOT, ScOTConfig
    from utils import resolve_runtime_device

    paddle.seed(0)
    np.random.seed(0)
    paddle.set_device(resolve_runtime_device())

    scale = {**COMMON_SCALE_ARGS, **MODEL_MAP[args.model]}
    config = ScOTConfig(
        image_size=args.resolution,
        patch_size=scale["patch_size"],
        num_channels=args.num_channels,
        num_out_channels=args.num_out_channels,
        embed_dim=scale["embed_dim"],
        depths=scale["depths"],
        num_heads=scale["num_heads"],
        skip_connections=scale["skip_connections"],
        window_size=scale["window_size"],
        mlp_ratio=scale["mlp_ratio"],
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=[0, args.num_out_channels],
        residual_model="convnext",
        use_conditioning=args.time_conditioning,
        learn_residual=False,
    )

    build_start = time.perf_counter()
    model = ScOT(config)
    # Same wrapping train.py:340 applies, so the benchmark measures the training path.
    if args.worker_mode == "cinn":
        model = paddle.jit.to_static(model, full_graph=True)
    model.train()

    # Mirrors trainer.py: AdamW + global-norm clipping at max_grad_norm=5.0.
    optimizer = paddle.optimizer.AdamW(
        learning_rate=5e-5,
        parameters=model.parameters(),
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-6,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0),
    )
    build_seconds = time.perf_counter() - build_start

    # One fixed synthetic batch: isolates compute from data loading, and makes the
    # loss trace comparable across modes (a divergence signals a to_static bug,
    # not a speed problem).
    shape = [args.batch_size, args.num_channels, args.resolution, args.resolution]
    pixel_values = paddle.to_tensor(np.random.rand(*shape).astype("float32"))
    label_shape = [args.batch_size, args.num_out_channels, args.resolution, args.resolution]
    labels = paddle.to_tensor(np.random.rand(*label_shape).astype("float32"))
    inputs = {"pixel_values": pixel_values, "labels": labels}
    if args.time_conditioning:
        inputs["time"] = paddle.to_tensor(
            np.full([args.batch_size], 0.5, dtype="float32")
        )

    def train_step():
        # Mirrors trainer.py:530 — under to_static the ScOTOutput dataclass carries
        # pir Values that cannot backward, so the CINN path takes the eager tuple.
        if args.worker_mode == "cinn":
            loss = model(**inputs, return_dict=False)[0]
        else:
            loss = model(**inputs).loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return float(loss)

    warmup_start = time.perf_counter()
    warmup_losses = [train_step() for _ in range(args.warmup)]
    paddle.device.synchronize()
    warmup_seconds = time.perf_counter() - warmup_start

    step_times_ms = []
    losses = []
    for _ in range(args.steps):
        step_start = time.perf_counter()
        losses.append(train_step())
        paddle.device.synchronize()
        step_times_ms.append((time.perf_counter() - step_start) * 1000.0)

    result = {
        "mode": args.worker_mode,
        "build_seconds": build_seconds,
        # For CINN the warm-up window absorbs the one-time compilation cost.
        "warmup_seconds": warmup_seconds,
        "median_step_ms": statistics.median(step_times_ms),
        "mean_step_ms": statistics.fmean(step_times_ms),
        "min_step_ms": min(step_times_ms),
        "max_step_ms": max(step_times_ms),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "warmup_first_loss": warmup_losses[0] if warmup_losses else None,
    }
    if args.result_path:
        with open(args.result_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream)
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- driver


def run_mode(mode, args, script_dir):
    result_path = os.path.join(script_dir, f".benchmark_{mode}.json")
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker_mode", mode,
        "--result_path", result_path,
        "--model", args.model,
        "--batch_size", str(args.batch_size),
        "--resolution", str(args.resolution),
        "--num_channels", str(args.num_channels),
        "--num_out_channels", str(args.num_out_channels),
        "--steps", str(args.steps),
        "--warmup", str(args.warmup),
    ]
    if args.time_conditioning:
        cmd.append("--time_conditioning")

    env = dict(os.environ)
    env[MODE_ENV] = "1" if mode == "cinn" else "0"
    env["WANDB_MODE"] = "disabled"

    print(f"\n=== running mode: {mode} ({MODE_ENV}={env[MODE_ENV]}) ===", flush=True)
    wall_start = time.perf_counter()
    completed = subprocess.run(cmd, env=env, cwd=os.path.dirname(script_dir))
    wall_seconds = time.perf_counter() - wall_start
    if completed.returncode != 0:
        raise SystemExit(f"mode '{mode}' failed with exit code {completed.returncode}")

    with open(result_path, "r", encoding="utf-8") as stream:
        result = json.load(stream)
    os.remove(result_path)
    result["wall_seconds"] = wall_seconds
    return result


def report(results, args):
    print("\n" + "=" * 72)
    print(
        f"ScOT-{args.model}  batch={args.batch_size}  resolution={args.resolution}  "
        f"warmup={args.warmup}  timed_steps={args.steps}"
    )
    print("=" * 72)
    for result in results:
        print(
            f"{result['mode']:>8}: median {result['median_step_ms']:8.1f} ms/step  "
            f"mean {result['mean_step_ms']:8.1f}  "
            f"min {result['min_step_ms']:8.1f}  max {result['max_step_ms']:8.1f}  "
            f"| warmup {result['warmup_seconds']:7.1f} s  total {result['wall_seconds']:7.1f} s"
        )

    by_mode = {result["mode"]: result for result in results}
    if "dygraph" in by_mode and "cinn" in by_mode:
        baseline = by_mode["dygraph"]["median_step_ms"]
        compiled = by_mode["cinn"]["median_step_ms"]
        speedup = (baseline - compiled) / baseline * 100.0
        verdict = "faster" if speedup > 0 else "SLOWER"
        print(
            f"\nsteady-state: CINN is {abs(speedup):.1f}% {verdict} "
            f"({baseline:.1f} -> {compiled:.1f} ms/step)"
        )

        # Compilation only pays off past this many steps.
        overhead = by_mode["cinn"]["warmup_seconds"] - by_mode["dygraph"]["warmup_seconds"]
        gain_per_step = (baseline - compiled) / 1000.0
        if gain_per_step > 0 and overhead > 0:
            print(
                f"break-even: ~{int(overhead / gain_per_step)} steps "
                f"(one-time cost {overhead:.1f} s vs {gain_per_step * 1000:.1f} ms saved/step)"
            )

        loss_gap = abs(by_mode["cinn"]["first_loss"] - by_mode["dygraph"]["first_loss"])
        scale = max(abs(by_mode["dygraph"]["first_loss"]), 1e-12)
        print(
            f"loss check: first timed-step loss dygraph {by_mode['dygraph']['first_loss']:.6f} "
            f"vs cinn {by_mode['cinn']['first_loss']:.6f} "
            f"(relative gap {loss_gap / scale:.2e})"
        )
        if loss_gap / scale > 1e-2:
            print(
                "WARNING: losses diverge by more than 1%. to_static may be numerically "
                "wrong here — investigate before trusting the speedup."
            )


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.worker_mode is not None:
        run_worker(args)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
        unknown = [mode for mode in modes if mode not in ("dygraph", "cinn")]
        if unknown:
            raise SystemExit(f"unknown modes: {unknown}")
        report([run_mode(mode, args, script_dir) for mode in modes], args)
