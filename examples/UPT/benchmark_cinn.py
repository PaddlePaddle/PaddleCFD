#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


UPT_DIR = Path(__file__).resolve().parent
PADDLECFD_ROOT = UPT_DIR.parents[1]
COMPILER_ENV_KEYS = (
    "FLAGS_enable_pir_api",
    "FLAGS_prim_enable_dynamic",
    "FLAGS_prim_all",
    "FLAGS_use_cinn",
    "FLAGS_print_ir",
    "ENABLE_FALL_BACK",
    "STRICT_MODE",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run matched dynamic and CINN UPT training benchmarks."
    )
    parser.add_argument("--hp", required=True, help="UPT training YAML")
    parser.add_argument("--device", default="0", help="single GPU device id")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="epoch timing entries excluded from steady-state metrics",
    )
    parser.add_argument(
        "--mindurationrun",
        action="store_true",
        help="limit both runs with UPT's full-model minimum-duration mode",
    )
    parser.add_argument("--output", help="output JSON path")
    return parser.parse_args()


def make_env(mode, device):
    env = os.environ.copy()
    for key in COMPILER_ENV_KEYS:
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    env["PYTHONUNBUFFERED"] = "1"
    if mode == "cinn":
        env.update(
            {
                "FLAGS_enable_pir_api": "true",
                "FLAGS_prim_enable_dynamic": "true",
                "FLAGS_prim_all": "true",
                "FLAGS_use_cinn": "true",
                "FLAGS_print_ir": "false",
                "ENABLE_FALL_BACK": "1",
                "STRICT_MODE": "0",
            }
        )
    return env


def load_yaml(path):
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def collect_metrics(mode, log_path, wall_time, compiler_marker, warmup_epochs):
    run_dir = log_path.parent
    summary = load_yaml(run_dir / "primitive" / "summary.yaml")
    entries = load_yaml(run_dir / "primitive" / "entries.yaml")
    config = load_yaml(run_dir / "primitive" / "config.yaml")

    epoch_times = entries["profiling/train_update_time/0/epoch"]
    ordered_times = [value for _, value in sorted(epoch_times.items())]
    steady_times = ordered_times[warmup_epochs:]
    if not steady_times:
        raise RuntimeError(
            f"{mode} produced {len(ordered_times)} epoch timing entries, "
            f"which is not enough for warmup_epochs={warmup_epochs}"
        )

    effective_batch_size = config["trainer"]["effective_batch_size"]
    steady_update_time = statistics.mean(steady_times)
    return {
        "mode": mode,
        "stage_id": run_dir.name,
        "log_path": str(log_path),
        "compiler_marker_found": compiler_marker,
        "wall_time_seconds": wall_time,
        "train_time_seconds": summary.get("profiler/train"),
        "update_time_seconds": summary.get("profiler/train/update"),
        "steady_update_seconds": steady_update_time,
        "throughput_samples_per_second": effective_batch_size / steady_update_time,
        "final_train_loss": summary.get("loss/online/total/E1"),
        "best_test_loss": summary.get("loss/test/total/min"),
        "epoch_update_seconds": ordered_times,
    }


def run_mode(mode, args, result_dir, run_stamp):
    command = [
        args.python,
        "main_train.py",
        "--accelerator",
        "gpu",
        "--devices",
        str(args.device),
        "--wandb_mode",
        "disabled",
        "--cuda_profiling",
        "--compiler",
        mode,
        "--name",
        f"benchmark_{mode}_{run_stamp}",
        "--hp",
        args.hp,
    ]
    if args.mindurationrun:
        command.append("--mindurationrun")

    print(f"\n[{mode}] {shlex.join(command)}", flush=True)
    console_path = result_dir / f"{mode}.console.log"
    log_path = None
    compiler_marker = False
    started_at = time.perf_counter()
    with console_path.open("w", encoding="utf-8") as console_file:
        process = subprocess.Popen(
            command,
            cwd=UPT_DIR,
            env=make_env(mode, args.device),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(f"[{mode}] {line}", end="", flush=True)
            console_file.write(line)
            match = re.search(r"log file:\s*(.+)$", line)
            if match:
                candidate = Path(match.group(1).strip())
                log_path = (
                    candidate if candidate.is_absolute() else (UPT_DIR / candidate)
                ).resolve()
            if "build_cinn_pass" in line or "FusionOp count" in line:
                compiler_marker = True
        return_code = process.wait()
    wall_time = time.perf_counter() - started_at

    if return_code != 0:
        raise RuntimeError(
            f"{mode} training failed with exit code {return_code}; "
            f"see {console_path}"
        )
    if log_path is None or not log_path.exists():
        raise RuntimeError(f"could not locate the {mode} training log")

    result = collect_metrics(
        mode=mode,
        log_path=log_path,
        wall_time=wall_time,
        compiler_marker=compiler_marker,
        warmup_epochs=args.warmup_epochs,
    )
    result["command"] = command
    result["console_path"] = str(console_path)
    return result


def print_report(dynamic, cinn):
    speedup = dynamic["steady_update_seconds"] / cinn["steady_update_seconds"] - 1
    throughput_gain = (
        cinn["throughput_samples_per_second"]
        / dynamic["throughput_samples_per_second"]
        - 1
    )
    print("\nUPT compiler benchmark")
    print("mode     wall(s)  steady update(s)  samples/s  best test loss")
    for result in (dynamic, cinn):
        print(
            f"{result['mode']:<8} "
            f"{result['wall_time_seconds']:>7.2f} "
            f"{result['steady_update_seconds']:>17.4f} "
            f"{result['throughput_samples_per_second']:>10.2f} "
            f"{result['best_test_loss']!s:>15}"
        )
    print(f"steady update speedup: {speedup * 100:.2f}%")
    print(f"steady throughput gain: {throughput_gain * 100:.2f}%")
    print(f"CINN compiler log marker found: {cinn['compiler_marker_found']}")


def main():
    args = parse_args()
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")

    hp_path = Path(args.hp)
    if not hp_path.is_absolute():
        hp_path = UPT_DIR / hp_path
    if not hp_path.exists():
        raise FileNotFoundError(f"training YAML does not exist: {hp_path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = PADDLECFD_ROOT / "output" / "UPT" / "benchmarks" / run_stamp
    result_dir.mkdir(parents=True, exist_ok=False)

    dynamic = run_mode("none", args, result_dir, run_stamp)
    cinn = run_mode("cinn", args, result_dir, run_stamp)
    print_report(dynamic, cinn)

    report = {
        "dynamic": dynamic,
        "cinn": cinn,
        "steady_update_speedup_percent": (
            dynamic["steady_update_seconds"] / cinn["steady_update_seconds"] - 1
        )
        * 100,
        "steady_throughput_gain_percent": (
            cinn["throughput_samples_per_second"]
            / dynamic["throughput_samples_per_second"]
            - 1
        )
        * 100,
    }
    output_path = Path(args.output) if args.output else result_dir / "result.json"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"result: {output_path.resolve()}")


if __name__ == "__main__":
    main()
