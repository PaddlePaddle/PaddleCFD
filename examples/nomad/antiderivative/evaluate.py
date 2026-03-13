import argparse
import numpy as np

from data_utils import generate_data


def compute_l2_error(s_true, s_pred):

    num_samples = s_true.shape[0]

    errors = []

    for i in range(num_samples):

        e = np.linalg.norm(
            s_true[i, :, 0] - s_pred[i, :, 0], 2
        ) / np.linalg.norm(
            s_true[i, :, 0], 2
        )

        errors.append(e)

    return np.array(errors)


def main(n, decoder):

    P = 500
    m = 500
    num_test = 1000

    # 与 inference 保持一致
    rng = np.random.default_rng(123)

    freqs = rng.uniform(0, 10, size=(num_test,))

    _, _, s_test = generate_data(freqs, m, P)

    s_test = s_test[..., None]

    # 加载推理结果
    s_pred = np.load("results/pred.npy")

    if s_pred.shape != s_test.shape:
        raise ValueError(
            f"Shape mismatch: pred {s_pred.shape}, gt {s_test.shape}"
        )

    errors = compute_l2_error(s_test, s_pred)

    print(
        "The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"
        % (
            np.mean(errors),
            np.std(errors),
            np.min(errors),
            np.max(errors),
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("n", type=int)
    parser.add_argument("decoder", type=str)

    args = parser.parse_args()

    main(args.n, args.decoder)