from typing import Iterable

import numpy as np
import paddle
import xarray as xr
import xskillscore as xs


class EnsembleMetrics:
    def __init__(
        self,
        per_model: bool = False,
        mean_over_samples: bool = True,
        metric_fns: Iterable[str] = ["mse"],
    ):
        self.per_model = per_model
        self.mean_over_samples = mean_over_samples

        try:
            unique_metrics = set(metric_fns)
        except TypeError:
            raise TypeError("metric_fns must be an iterable") from None

        allowed = {"mse", "crps", "ssr", "nll"}
        invalid = unique_metrics - allowed

        if not isinstance(metric_fns, Iterable) or isinstance(metric_fns, (str, bytes)):
            raise TypeError(f"metric_fns must be iterable of strings, got {type(metric_fns)}")
        if len(unique_metrics) == 0:
            raise ValueError("At least one metric must be specified")
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}. Allowed: {allowed}")

        self.metric_fns = list(unique_metrics)

    def mse(self, preds, targets, mean_axis):
        mse_elementwise = paddle.nn.functional.mse_loss(preds, targets, reduction="none")
        mse = paddle.mean(mse_elementwise, axis=mean_axis)
        return mse

    def rmse(self, preds, targets, mean_axis):
        return paddle.sqrt(self.mse(preds, targets, mean_axis))

    def crps(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        member_dim: str = "member",
    ) -> float | np.ndarray:
        dummy_dims = [f"dummy_dim_{i}" for i in range(targets.ndim - 1)]
        preds_da = xr.DataArray(preds, dims=[member_dim, "sample"] + dummy_dims)
        targets_da = xr.DataArray(targets, dims=["sample"] + dummy_dims)
        mean_dims = ["sample"] + dummy_dims if self.mean_over_samples else dummy_dims
        crps = xs.crps_ensemble(
            observations=targets_da, forecasts=preds_da, member_dim=member_dim, dim=mean_dims
        ).values  # shape: ()
        return paddle.to_tensor(crps)

    def ssr(self, preds, targets, skill_metric: float = None, mean_axis=None):
        variance = paddle.var(preds, axis=0).mean(axis=mean_axis)
        spread = paddle.sqrt(variance)
        skill_metric = (
            self.rmse(preds.mean(axis=0), targets, mean_axis=mean_axis) if skill_metric is None else skill_metric
        )
        return spread / skill_metric

    def nll(preds, targets, mean_dims=None, var_correction: int = 1):
        """Compute the negative log-likelihood of an ensemble of predictions.

        Args:
            preds (paddle.Tensor): predictions.
            targets (paddle.Tensor): targets.
            mean_dims (Tuple, optional): dims used to calculate the mean. Defaults to None.
            var_correction (int, optional): var_correction = 0 means sample variance, 1 means unbiased variance, unbiased variance needs to be divided by (n_members - 1). Defaults to 1.
        """
        mean_preds = paddle.mean(preds, axis=0)
        variance = paddle.var(preds, axis=0, unbiased=(var_correction == 1))
        variance = paddle.clip(variance, min=1e-6)
        normal_dist = paddle.distribution.Normal(loc=mean_preds, scale=paddle.sqrt(variance))
        nll = -normal_dist.log_prob(targets)
        return paddle.mean(nll, axis=mean_dims)

    def metric(self, preds, targets, crps_member_dim: int = 0):
        assert (
            preds.shape[1] == targets.shape[0]
        ), f"preds.shape[1] ({preds.shape[1]}) != targets.shape[0] ({targets.shape[0]})"
        # shape could be: preds: (10, 730, 3, 60, 60), targets: (730, 3, 60, 60)
        n_preds, n_samples = preds.shape[:2]

        # Compute the mean prediction
        mean_preds = paddle.mean(preds, axis=0)
        mean_axis = tuple(range(0 if self.mean_over_samples else 1, mean_preds.ndim))

        metric_dict = {}
        # MSE
        if "mse" in self.metric_fns:
            mse = self.mse(mean_preds, targets, mean_axis)
            metric_dict["mse"] = mse
            # MSE pre model
            if self.per_model:
                # next, compute the MSE for each model
                mse_per = self.mse(preds, paddle.expand_as(targets.unsqueeze(0), preds), False)
                mse_per_mean = paddle.mean(mse_per)
                metric_dict.update({"mse_per": mse_per, "mse_per_mean": mse_per_mean})

        # CRPS
        if "crps" in self.metric_fns:
            crps = self.crps(preds.numpy(), targets.numpy(), crps_member_dim)
            metric_dict["crps"] = crps

        # SSR
        if "ssr" in self.metric_fns:
            skill_metric = paddle.sqrt(mse) if "mse" in self.metric_fns else None
            ssr = self.ssr(preds, targets, skill_metric=skill_metric, mean_axis=mean_axis)
            metric_dict["ssr"] = ssr

        # compute negative log-likelihood
        if "nll" in self.metric_fns:
            nll = self.nll(preds, targets, mean_axis=mean_axis)
            metric_dict["nll"] = nll
        return metric_dict
