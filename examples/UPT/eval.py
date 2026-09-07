# examples/UPT/eval.py
import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from copy import deepcopy
from pathlib import Path


MODE = "pressure mesh_pos sdf query_pos"


def import_runtime_deps():
    global np, paddle, yaml
    import numpy as np
    import paddle
    import yaml


def setup_imports():
    upt_dir = Path(__file__).resolve().parent
    root = upt_dir.parent.parent
    source_upt = root / "source" / "UPT"
    for path in (upt_dir, root, source_upt, source_upt / "KappaModules"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    if "models" not in sys.modules:
        models_dir = root / "ppcfd" / "models" / "UPT"
        spec = importlib.util.spec_from_file_location(
            "models",
            models_dir / "__init__.py",
            submodule_search_locations=[str(models_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["models"] = module
        spec.loader.exec_module(module)
    return root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ShapeNetCar UPT inference, optionally with split-level metrics."
    )
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--checkpoint", default="best_model.loss.test.total")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Single-sample .npz path. With --eval, this is the output directory "
            "for optional per-sample predictions."
        ),
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a range of samples and write aggregate metrics.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="First sample index used by --eval.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples used by --eval. Defaults to the rest of the split.",
    )
    parser.add_argument(
        "--metrics_output",
        type=Path,
        help="JSON path for aggregate metrics. Defaults under run_dir/inference.",
    )
    parser.add_argument(
        "--per_sample_output",
        type=Path,
        help="CSV path for per-sample metrics. Defaults next to metrics_output.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="With --eval, also save per-sample prediction .npz files.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Progress print interval for --eval. Use 0 to disable progress prints.",
    )
    return parser.parse_args()


def sanitize_for_path(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_hp(run_dir):
    hp_uri = run_dir / "hp_resolved.yaml"
    if not hp_uri.exists():
        raise FileNotFoundError(f"missing hp_resolved.yaml: {hp_uri}")
    with hp_uri.open() as f:
        hp = yaml.safe_load(f)
    if hp["model"]["kind"] != "rans_simformer_nognn_sdf_model":
        raise ValueError("This script only supports rans_simformer_nognn_sdf_model.")
    return hp


def build_dataset(root, hp, split):
    from datasets import dataset_from_kwargs
    from providers.dataset_config_provider import DatasetConfigProvider

    dataset_provider = DatasetConfigProvider(
        global_dataset_paths={"shapenet_car": root},
        local_dataset_path=None,
        data_source_modes=None,
    )
    return dataset_from_kwargs(
        dataset_config_provider=dataset_provider,
        **deepcopy(hp["datasets"][split]),
    )


def build_model(hp, data_container, device):
    from models import model_from_kwargs

    model = model_from_kwargs(
        **deepcopy(hp["model"]),
        input_shape=(None, 3),
        output_shape=(None, 1),
        data_container=data_container,
    ).to(device)
    return model


def load_checkpoint(model, run_dir, checkpoint):
    checkpoint_dir = run_dir / "checkpoints"
    missing = []
    for name, submodel in model.submodels.items():
        filename = f"{model.name}.{name} cp={checkpoint} model.th"
        uri = checkpoint_dir / filename
        if not uri.exists():
            missing.append(uri.name)
            continue
        state = paddle.load(str(uri))
        submodel.set_state_dict(state.get("state_dict", state))

    if missing:
        available = sorted(path.name for path in checkpoint_dir.glob("* model.th"))
        preview = "\n".join(f"  - {name}" for name in available[:20])
        if len(available) > 20:
            preview += f"\n  ... {len(available) - 20} more"
        raise FileNotFoundError(
            "missing checkpoint files:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + ("\navailable checkpoints:\n" + preview if available else "")
        )
    model.eval()


def make_data(model, data_container, split):
    wrapped_dataset, collator = data_container.get_dataset(split, mode=MODE)
    if collator is None:
        raise RuntimeError(f"dataset split '{split}' has no collator for mode '{MODE}'")
    return wrapped_dataset, collator


def batch_item(batch, name, device):
    from kappadata.wrappers import ModeWrapper

    return ModeWrapper.get_item(mode=MODE, item=name, batch=batch).to(device)


def denormalize(tensor, dataset, device):
    return tensor * dataset.std.to(device) + dataset.mean.to(device)


def infer_sample(model, dataset, wrapped_dataset, collator, sample_idx, device):
    batch, ctx = collator([wrapped_dataset[sample_idx]])

    with paddle.no_grad():
        prediction = model(
            mesh_pos=batch_item(batch, "mesh_pos", device),
            sdf=batch_item(batch, "sdf", device),
            query_pos=batch_item(batch, "query_pos", device),
            batch_idx=ctx["batch_idx"].to(device),
            unbatch_idx=ctx["unbatch_idx"].to(device),
            unbatch_select=ctx["unbatch_select"].to(device),
        )["x_hat"].squeeze(1)

    target = batch_item(batch, "pressure", device).squeeze(1)
    prediction = denormalize(prediction, dataset, device)
    target = denormalize(target, dataset, device)
    query_pos = batch_item(batch, "query_pos", device)[0, : len(prediction)]
    return prediction, target, query_pos


def compute_metrics(prediction, target):
    error = prediction - target
    abs_error = paddle.abs(error)
    sq_error_sum = float(paddle.sum(error**2).item())
    abs_error_sum = float(paddle.sum(abs_error).item())
    target_sq_sum = float(paddle.sum(target**2).item())
    max_abs_error = float(paddle.max(abs_error).item())
    num_points = int(np.prod(error.shape))
    mse = sq_error_sum / max(num_points, 1)
    mae = abs_error_sum / max(num_points, 1)
    relative_l2 = math.sqrt(sq_error_sum) / max(math.sqrt(target_sq_sum), 1e-12)
    return {
        "num_points": num_points,
        "sq_error_sum": sq_error_sum,
        "abs_error_sum": abs_error_sum,
        "target_sq_sum": target_sq_sum,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "relative_l2": relative_l2,
        "max_abs_error": max_abs_error,
    }


def save_prediction_npz(
    output,
    query_pos,
    prediction,
    target,
    metrics,
    sample_idx,
    split,
    checkpoint,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    error = prediction - target
    np.savez(
        output,
        query_pos=query_pos.cpu().numpy(),
        prediction=prediction.cpu().numpy(),
        target=target.cpu().numpy(),
        abs_error=paddle.abs(error).cpu().numpy(),
        mse=metrics["mse"],
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        relative_l2=metrics["relative_l2"],
        max_abs_error=metrics["max_abs_error"],
        sample_idx=sample_idx,
        split=split,
        checkpoint=checkpoint,
    )


def resolve_single_output(args):
    default_uri = (
        args.run_dir / "inference" / f"{args.split}_{args.sample_idx:06d}.npz"
    )
    if args.output is None:
        return default_uri
    if args.output.suffix == ".npz":
        return args.output
    return args.output / default_uri.name


def resolve_eval_outputs(args):
    checkpoint_tag = sanitize_for_path(args.checkpoint)
    eval_dir = args.output or args.run_dir / "inference" / f"{args.split}_{checkpoint_tag}"
    metrics_uri = (
        args.metrics_output
        or eval_dir / f"metrics_{args.split}_{checkpoint_tag}.json"
    )
    per_sample_uri = (
        args.per_sample_output
        or metrics_uri.with_name(metrics_uri.stem + "_per_sample.csv")
    )
    pred_dir = eval_dir / "predictions"
    return eval_dir, metrics_uri, per_sample_uri, pred_dir


def resolve_eval_indices(args, dataset_len):
    if not 0 <= args.start_idx < dataset_len:
        raise IndexError(f"start_idx must be in [0, {dataset_len - 1}]")
    end_idx = dataset_len
    if args.num_samples is not None:
        if args.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        end_idx = min(dataset_len, args.start_idx + args.num_samples)
    return list(range(args.start_idx, end_idx))


def summarize_records(records, elapsed_sec, args):
    total_points = sum(row["num_points"] for row in records)
    total_sq = sum(row["sq_error_sum"] for row in records)
    total_abs = sum(row["abs_error_sum"] for row in records)
    total_target_sq = sum(row["target_sq_sum"] for row in records)
    mse = total_sq / max(total_points, 1)
    relative_l2_values = [row["relative_l2"] for row in records]
    worst = max(records, key=lambda row: row["relative_l2"])
    best = min(records, key=lambda row: row["relative_l2"])
    return {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "start_idx": args.start_idx,
        "num_samples": len(records),
        "num_points": total_points,
        "elapsed_sec": elapsed_sec,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": total_abs / max(total_points, 1),
        "relative_l2_global": math.sqrt(total_sq) / max(math.sqrt(total_target_sq), 1e-12),
        "relative_l2_mean": float(np.mean(relative_l2_values)),
        "relative_l2_median": float(np.median(relative_l2_values)),
        "relative_l2_p95": float(np.percentile(relative_l2_values, 95)),
        "sample_mse_mean": float(np.mean([row["mse"] for row in records])),
        "sample_mae_mean": float(np.mean([row["mae"] for row in records])),
        "best_sample_idx": int(best["sample_idx"]),
        "best_sample_relative_l2": best["relative_l2"],
        "worst_sample_idx": int(worst["sample_idx"]),
        "worst_sample_relative_l2": worst["relative_l2"],
    }


def write_eval_outputs(metrics_uri, per_sample_uri, summary, records):
    metrics_uri.parent.mkdir(parents=True, exist_ok=True)
    with metrics_uri.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    public_fields = [
        "sample_idx",
        "num_points",
        "mse",
        "rmse",
        "mae",
        "relative_l2",
        "max_abs_error",
    ]
    per_sample_uri.parent.mkdir(parents=True, exist_ok=True)
    with per_sample_uri.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=public_fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in public_fields})


def run_single(args, model, dataset, wrapped_dataset, collator):
    if not 0 <= args.sample_idx < len(dataset):
        raise IndexError(f"sample_idx must be in [0, {len(dataset) - 1}]")

    prediction, target, query_pos = infer_sample(
        model=model,
        dataset=dataset,
        wrapped_dataset=wrapped_dataset,
        collator=collator,
        sample_idx=args.sample_idx,
        device=args.device,
    )
    metrics = compute_metrics(prediction, target)
    output = resolve_single_output(args)
    save_prediction_npz(
        output=output,
        query_pos=query_pos,
        prediction=prediction,
        target=target,
        metrics=metrics,
        sample_idx=args.sample_idx,
        split=args.split,
        checkpoint=args.checkpoint,
    )
    print(f"Saved: {output}")
    print(
        "MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, "
        "relative_l2={relative_l2:.6f}, max_abs_error={max_abs_error:.6f}".format(
            **metrics
        )
    )


def run_eval(args, model, dataset, wrapped_dataset, collator):
    _, metrics_uri, per_sample_uri, pred_dir = resolve_eval_outputs(args)
    indices = resolve_eval_indices(args, len(dataset))
    records = []
    start_time = time.time()

    for ordinal, sample_idx in enumerate(indices, start=1):
        prediction, target, query_pos = infer_sample(
            model=model,
            dataset=dataset,
            wrapped_dataset=wrapped_dataset,
            collator=collator,
            sample_idx=sample_idx,
            device=args.device,
        )
        metrics = compute_metrics(prediction, target)
        metrics["sample_idx"] = sample_idx
        records.append(metrics)

        if args.save_predictions:
            save_prediction_npz(
                output=pred_dir / f"{args.split}_{sample_idx:06d}.npz",
                query_pos=query_pos,
                prediction=prediction,
                target=target,
                metrics=metrics,
                sample_idx=sample_idx,
                split=args.split,
                checkpoint=args.checkpoint,
            )

        if args.print_every and (ordinal == 1 or ordinal % args.print_every == 0):
            print(
                f"[{ordinal}/{len(indices)}] sample_idx={sample_idx} "
                f"mse={metrics['mse']:.6f} relative_l2={metrics['relative_l2']:.6f}"
            )

    elapsed_sec = time.time() - start_time
    summary = summarize_records(records, elapsed_sec, args)
    write_eval_outputs(metrics_uri, per_sample_uri, summary, records)
    print(f"Saved metrics: {metrics_uri}")
    print(f"Saved per-sample metrics: {per_sample_uri}")
    if args.save_predictions:
        print(f"Saved predictions: {pred_dir}")
    print(
        "Eval {split}: samples={num_samples}, points={num_points}, "
        "MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, "
        "relative_l2_global={relative_l2_global:.6f}, "
        "relative_l2_mean={relative_l2_mean:.6f}".format(**summary)
    )


def main():
    args = parse_args()
    import_runtime_deps()
    root = setup_imports()
    import paddle_utils  # noqa: F401

    from utils.data_container import DataContainer

    args.run_dir = args.run_dir.resolve()
    hp = load_hp(args.run_dir)
    paddle.set_device(args.device)

    dataset = build_dataset(root=root, hp=hp, split=args.split)
    data_container = DataContainer(**{args.split: dataset})
    model = build_model(hp=hp, data_container=data_container, device=args.device)
    load_checkpoint(model=model, run_dir=args.run_dir, checkpoint=args.checkpoint)
    wrapped_dataset, collator = make_data(
        model=model, data_container=data_container, split=args.split
    )

    if args.eval:
        run_eval(
            args=args,
            model=model,
            dataset=dataset,
            wrapped_dataset=wrapped_dataset,
            collator=collator,
        )
    else:
        run_single(
            args=args,
            model=model,
            dataset=dataset,
            wrapped_dataset=wrapped_dataset,
            collator=collator,
        )


if __name__ == "__main__":
    main()
