import argparse
import json
from pathlib import Path

import numpy as np

from antiderivative.data_utils import generate_data


EXAMPLE_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "NOMAD"


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def pass_diff(paddle_mean, reference_mean, tolerance):
    diff = float(abs(paddle_mean - reference_mean))
    rel = float(diff / max(abs(reference_mean), 1e-12))
    return diff, rel, bool(diff <= tolerance)


def antiderivative_metric(n, decoder, tolerance, reference_root):
    pred = np.load(EXAMPLE_ROOT / "antiderivative/results/pred.npy")
    rng = np.random.default_rng(12345)
    freqs = rng.uniform(0, 10, size=(pred.shape[0],))
    _, _, target = generate_data(freqs, 500, 500)
    target = target[..., None]
    errors = [
        np.linalg.norm(target[i, :, 0] - pred[i, :, 0], 2)
        / np.linalg.norm(target[i, :, 0], 2)
        for i in range(pred.shape[0])
    ]

    ref_path = (
        reference_root
        / "Antiderivative/Error_Vectors"
        / f"Error_Antiderivative_DeepONet_nhat{n}_{decoder}.npz"
    )
    ref_errors = np.load(ref_path)["test_error"]
    diff, rel, passed = pass_diff(np.mean(errors), np.mean(ref_errors), tolerance)
    return {
        "case": "antiderivative",
        "paddle": stats(errors),
        "reference": stats(ref_errors),
        "mean_abs_diff": diff,
        "mean_rel_diff": rel,
        "pass_0.2_percent": passed,
        "note": "Paddle script uses its local synthetic data generator; compare with caution unless regenerated from the original JAX data pipeline.",
    }


def advection_metric(n, decoder, tolerance, reference_root):
    pred = np.load(EXAMPLE_ROOT / "advection/results/prediction.npz")["prediction"]
    data = np.load(EXAMPLE_ROOT / "advection/pure_advection_traintest.npz")
    target = data["solution"][-pred.shape[0] :].reshape(pred.shape[0], 25600, 1)
    errors = [
        np.linalg.norm(target[i, :, 0] - pred[i, :, 0], 2)
        / np.linalg.norm(target[i, :, 0], 2)
        for i in range(pred.shape[0])
    ]

    ref_path = (
        reference_root
        / "Advection/Error_Vectors"
        / f"Error_Advection_DeepONet_nhat{n}_iteration0_{decoder}.npz"
    )
    ref_errors = np.load(ref_path)["test_error"]
    diff, rel, passed = pass_diff(np.mean(errors), np.mean(ref_errors), tolerance)
    return {
        "case": "advection",
        "paddle": stats(errors),
        "reference": stats(ref_errors),
        "mean_abs_diff": diff,
        "mean_rel_diff": rel,
        "pass_0.2_percent": passed,
        "paddle_num_test": int(pred.shape[0]),
        "reference_num_test": int(ref_errors.shape[-1]),
    }


def shallowwater_metric(n, decoder, tolerance, reference_root):
    pred = np.load(EXAMPLE_ROOT / "shallowwater/results/pred_test.npy")
    data = np.load(EXAMPLE_ROOT / "shallowwater/test_SW.npz")
    target = data["S_test"].reshape(pred.shape[0], 5 * 32 * 32, 3)

    paddle_errors = []
    for channel in range(3):
        paddle_errors.append(
            [
                np.linalg.norm(target[i, :, channel] - pred[i, :, channel], 2)
                / np.linalg.norm(target[i, :, channel], 2)
                for i in range(pred.shape[0])
            ]
        )

    ref_path = (
        reference_root
        / "ShallowWater/Error_vectors"
        / f"Error_SW_DeepONet_nhat{n}_iteration0_{decoder}.npz"
    )
    ref_errors = np.load(ref_path)["test_error"]
    channel_names = ["rho", "u", "v"]
    channels = []
    for idx, name in enumerate(channel_names):
        diff, rel, passed = pass_diff(
            np.mean(paddle_errors[idx]), np.mean(ref_errors[idx]), tolerance
        )
        channels.append(
            {
                "name": name,
                "paddle": stats(paddle_errors[idx]),
                "reference": stats(ref_errors[idx]),
                "mean_abs_diff": diff,
                "mean_rel_diff": rel,
                "pass_0.2_percent": passed,
            }
        )

    return {"case": "shallowwater", "channels": channels}


def main():
    parser = argparse.ArgumentParser(description="Compare NOMAD Paddle metrics with original reference error vectors.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--decoder", choices=["linear", "nonlinear"], default="nonlinear")
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=Path("metric_report.json"))
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Root of the NOMAD Google Drive package containing Error_Vectors and Data.",
    )
    args = parser.parse_args()

    report = [
        antiderivative_metric(args.n, args.decoder, args.tolerance, args.reference_root),
        advection_metric(args.n, args.decoder, args.tolerance, args.reference_root),
        shallowwater_metric(args.n, args.decoder, args.tolerance, args.reference_root),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
