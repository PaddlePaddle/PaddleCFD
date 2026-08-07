import argparse
from pathlib import Path

import numpy as np


def parse_csv_ints(raw_value):
    if raw_value is None or raw_value == "":
        return []
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def normalize_shape(shape):
    if len(shape) != 5:
        raise ValueError(f"Expected shape (T,B,C,H,W), got {shape}")
    return tuple(int(part) for part in shape)


def build_case_payload(shape, labels, bcs, seed):
    t, batch_size, channels, height, width = normalize_shape(shape)
    if not labels:
        raise ValueError("labels must not be empty")
    if len(bcs) != 2:
        raise ValueError("bcs must contain exactly 2 integers")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t, batch_size, channels, height, width), dtype=np.float32)
    state_labels = np.tile(np.asarray(labels, dtype=np.int64), (batch_size, 1))
    bcs_array = np.tile(np.asarray(bcs, dtype=np.int64), (batch_size, 1))
    return {
        "x": x.astype(np.float32, copy=False),
        "state_labels": state_labels,
        "bcs": bcs_array,
    }


def save_case_payload(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--shape", default="4,1,3,64,64", type=str)
    parser.add_argument("--labels", required=True, type=str)
    parser.add_argument("--bcs", default="0,0", type=str)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    payload = build_case_payload(
        shape=parse_csv_ints(args.shape),
        labels=parse_csv_ints(args.labels),
        bcs=parse_csv_ints(args.bcs),
        seed=args.seed,
    )
    save_case_payload(args.output, payload)
    print(f"Saved forward case to {args.output}")


if __name__ == "__main__":
    main()
