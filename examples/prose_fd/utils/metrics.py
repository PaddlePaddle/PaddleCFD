from collections import defaultdict

import numpy as np
import paddle
import scipy


def compute_metrics(output, label, mask=None, metrics=["_mse"], batched=False):
    """
    output, label: torch tensor (seq_len, ... output_dim) if batched=False
                                (bs, seq_len, ..., output_dim) if batched=True
    """
    assert (
        label.shape == output.shape
    ), f"shape mismatch: output: {output.shape}, label: {label.shape}"
    results = dict()
    seq_len = output.size(1) if batched else output.size(0)
    eps = 1e-07
    if metrics == "":
        return results
    for metric in metrics:
        if metric == "_mse":
            if batched:
                mse = ((output - label) ** 2).flatten(1).mean(1).tolist()
            else:
                mse = ((output - label) ** 2).mean().item()
            results[metric] = mse
        elif metric == "_rmse":
            if batched:
                mse = paddle.sqrt(((output - label) ** 2).flatten(1).mean(1)).tolist()
            else:
                mse = paddle.sqrt(((output - label) ** 2).mean()).item()
            results[metric] = mse
        elif metric.startswith("_l2_error"):
            if batched:
                if metric == "_l2_error":
                    predicted, true = output, label
                elif metric == "_l2_error_first_half":
                    predicted = output[:, : seq_len // 2]
                    true = label[:, : seq_len // 2]
                elif metric == "_l2_error_second_half":
                    predicted = output[:, seq_len // 2 :]
                    true = label[:, seq_len // 2 :]
                elif metric == "_l2_error_step_1":
                    predicted = output[:, 0:1]
                    true = label[:, 0:1]
                elif metric == "_l2_error_step_5":
                    if label.size(1) < 5:
                        predicted = output[:, 4:5]
                        true = label[:, 4:5]
                    else:
                        predicted = output[:, :5]
                        true = label[:, :5]
                elif metric == "_l2_error_step_10":
                    if label.size(1) < 10:
                        predicted = output[:, 9:10]
                        true = label[:, 9:10]
                    else:
                        predicted = output[:, :10]
                        true = label[:, :10]
                elif metric == "_l2_error_int":
                    predicted = output * mask
                    true = label * mask
                else:
                    assert False, f"Unknown metric: {metric}"
                error = paddle.sqrt(((true - predicted) ** 2).flatten(2).sum(2))
                scale = eps + paddle.sqrt((true**2).flatten(2).sum(2))
                error = paddle.div(error, scale).mean(dim=1).tolist()
            else:
                if metric == "_l2_error":
                    predicted, true = output, label
                elif metric == "_l2_error_first_half":
                    predicted = output[: seq_len // 2]
                    true = label[: seq_len // 2]
                elif metric == "_l2_error_second_half":
                    predicted = output[seq_len // 2 :]
                    true = label[seq_len // 2 :]
                elif metric == "_l2_error_step_1":
                    predicted = output[0:1]
                    true = label[0:1]
                elif metric == "_l2_error_step_5":
                    if label.size(0) < 5:
                        predicted = output[4:5]
                        true = label[4:5]
                    else:
                        predicted = output[:5]
                        true = label[:5]
                elif metric == "_l2_error_step_10":
                    if label.size(0) < 10:
                        predicted = output[9:10]
                        true = label[9:10]
                    else:
                        predicted = output[:10]
                        true = label[:10]
                elif metric == "_l2_error_int":
                    predicted = output * mask
                    true = label * mask
                else:
                    assert False, f"Unknown metric: {metric}"
                error = paddle.sqrt(((true - predicted) ** 2).flatten(1).sum(1))
                scale = eps + paddle.sqrt((true**2).flatten(1).sum(1))
                error = paddle.div(error, scale).mean().item()
            results[metric] = error
    return results
