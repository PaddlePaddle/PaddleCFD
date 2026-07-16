import argparse
import json
import statistics
import time
from pathlib import Path

import paddle

from ppcfd.models.nomad.advection.operator_model import OperatorModel as AdvectionModel
from ppcfd.models.nomad.antiderivative.nomad_antiderivative import (
    OperatorModel as AntiderivativeModel,
)
from ppcfd.models.nomad.shallowwater.nomad_model import OperatorModel as ShallowWaterModel


CASE_CONFIGS = {
    "antiderivative": {
        "model": AntiderivativeModel,
        "checkpoint": "antiderivative/checkpoints/nomad_antiderivative.pdparams",
        "m": 500,
        "p": 500,
        "du": 1,
        "dy": 1,
        "ds": 1,
    },
    "advection": {
        "model": AdvectionModel,
        "checkpoint": "advection/checkpoints/nomad_advection_model.pdparams",
        "m": 256,
        "p": 25600,
        "du": 1,
        "dy": 2,
        "ds": 1,
    },
    "shallowwater": {
        "model": ShallowWaterModel,
        "checkpoint": "shallowwater/checkpoints/nomad_sw_nonlinear_n10.pdparams",
        "m": 1024,
        "p": 5120,
        "du": 3,
        "dy": 3,
        "ds": 3,
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


def build_model(case, decoder, n):
    cfg = CASE_CONFIGS[case]
    branch_layers, trunk_layers = make_layers(cfg, n, decoder)
    if case == "shallowwater":
        model = cfg["model"](branch_layers, trunk_layers, n=n, decoder=decoder, ds=cfg["ds"])
    else:
        model = cfg["model"](
            branch_layers,
            trunk_layers,
            m=cfg["m"],
            P=cfg["p"],
            n=n,
            decoder=decoder,
            ds=cfg["ds"],
        )
    return model


def maybe_load_checkpoint(model, case, checkpoint):
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / CASE_CONFIGS[case]["checkpoint"]
    else:
        checkpoint = Path(checkpoint)
    if checkpoint.exists():
        model.set_state_dict(paddle.load(str(checkpoint)))
    return str(checkpoint)


def sync():
    if paddle.get_device().startswith("gpu"):
        paddle.device.synchronize()


def run_once(model, inputs):
    with paddle.no_grad():
        out = model(inputs)
    sync()
    return out


def benchmark(model, inputs, warmup, steps):
    for _ in range(warmup):
        run_once(model, inputs)
    start = time.perf_counter()
    for _ in range(steps):
        run_once(model, inputs)
    elapsed = time.perf_counter() - start
    return elapsed / steps


def summarize(values):
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
        }
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark NOMAD inference with Paddle compiler off/on.")
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), default="antiderivative")
    parser.add_argument("--decoder", choices=["linear", "nonlinear"], default="nonlinear")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", type=Path, default=Path("compiler_benchmark.json"))
    args = parser.parse_args()

    paddle.set_device(args.device)
    cfg = CASE_CONFIGS[args.case]
    points = args.points or cfg["p"]
    u = paddle.randn([args.batch_size, cfg["m"], cfg["du"]], dtype="float32")
    y = paddle.randn([args.batch_size, points, cfg["dy"]], dtype="float32")

    dynamic_model = build_model(args.case, args.decoder, args.n)
    checkpoint = maybe_load_checkpoint(dynamic_model, args.case, args.checkpoint)
    dynamic_model.eval()

    static_model = build_model(args.case, args.decoder, args.n)
    maybe_load_checkpoint(static_model, args.case, checkpoint)
    static_model.eval()
    static_model = paddle.jit.to_static(static_model, full_graph=True)

    dynamic_times = []
    static_times = []
    speedups = []
    for _ in range(args.repeats):
        dynamic_time = benchmark(dynamic_model, (u, y), args.warmup, args.steps)
        static_time = benchmark(static_model, (u, y), args.warmup, args.steps)
        dynamic_times.append(dynamic_time)
        static_times.append(static_time)
        speedups.append(dynamic_time / static_time if static_time > 0 else float("inf"))

    dynamic_time = statistics.fmean(dynamic_times)
    static_time = statistics.fmean(static_times)
    speedup = statistics.fmean(speedups)

    report = {
        "case": args.case,
        "decoder": args.decoder,
        "n": args.n,
        "device": args.device,
        "batch_size": args.batch_size,
        "points": points,
        "warmup": args.warmup,
        "steps": args.steps,
        "repeats": args.repeats,
        "checkpoint": checkpoint,
        "dynamic_avg_step_time_s": dynamic_time,
        "compiler_avg_step_time_s": static_time,
        "speedup": speedup,
        "speedup_percent": (speedup - 1.0) * 100.0,
        "dynamic_avg_step_time_s_runs": dynamic_times,
        "compiler_avg_step_time_s_runs": static_times,
        "speedup_runs": speedups,
        "dynamic_summary": summarize(dynamic_times),
        "compiler_summary": summarize(static_times),
        "speedup_summary": summarize(speedups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
