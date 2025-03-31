import paddle


class EnsembleMetrics:
    def __init__(self, per_model: bool = False, mean_over_samples: bool = True):
        self.per_model = per_model
        self.mean_over_samples = mean_over_samples

    def mse(self, preds, targets, mean_axis):
        mse_elementwise = paddle.nn.functional.mse_loss(preds, targets, reduction="none")
        mse = paddle.mean(mse_elementwise, axis=mean_axis)
        return mse

    def rmse(self, preds, targets, mean_axis):
        return paddle.sqrt(self.mse(preds, targets, mean_axis))

    def crps(self, preds, targets, member_dim: int = 0):
        assert (
            targets.ndim == preds.ndim - 1
        ), f"Error: the target dim {targets.ndim} should = pred dim {preds.ndim} - 1."

        # compute mean of |X - y|
        target_expanded = targets.unsqueeze(member_dim)
        abs_diff = paddle.abs(preds - target_expanded)
        mean_abs_diff = paddle.mean(abs_diff, axis=member_dim)

        # compute mean of |X - X'|
        preds_unsqueeze1 = paddle.unsqueeze(preds, axis=member_dim)
        preds_unsqueeze2 = paddle.unsqueeze(preds, axis=member_dim + 1)
        ensemble_diff = paddle.abs(preds_unsqueeze1 - preds_unsqueeze2)
        mean_ensemble_diff = paddle.mean(ensemble_diff, axis=(member_dim, member_dim + 1))

        # compute the CRPS
        crps = mean_abs_diff - 0.5 * mean_ensemble_diff

        # get mean of CRPS
        if self.mean_over_samples:
            return paddle.mean(crps)
        else:
            mean_dims = list(range(1, targets.ndim))
        return paddle.mean(crps, axis=mean_dims)

    def ssr(self, preds, targets, skill_metric: float = None, mean_axis=None):
        variance = paddle.var(preds, axis=0).mean(axis=mean_axis)
        spread = paddle.sqrt(variance)

        skill_metric = (
            self.rmse(preds.mean(axis=0), targets, mean_axis=mean_axis) if skill_metric is None else skill_metric
        )

        return spread / skill_metric

    def nll(preds, targets, mean_dims=None, var_correction: int = 1):
        """Compute the negative log-likelihood of an ensemble of predictions."""
        # var_correction = 0 表示样本方差，1 表示无偏方差, 无偏方差需除以(n_members - 1)
        mean_preds = paddle.mean(preds, axis=0)
        variance = paddle.var(preds, axis=0, unbiased=(var_correction == 1))
        variance = paddle.clip(variance, min=1e-6)
        normal_dist = paddle.distribution.Normal(loc=mean_preds, scale=paddle.sqrt(variance))
        nll = -normal_dist.log_prob(targets)
        return paddle.mean(nll, axis=mean_dims)

    def metric(self, preds, targets):
        assert (
            preds.shape[1] == targets.shape[0]
        ), f"preds.shape[1] ({preds.shape[1]}) != targets.shape[0] ({targets.shape[0]})"
        # shape could be: preds: (10, 730, 3, 60, 60), targets: (730, 3, 60, 60)
        n_preds, n_samples = preds.shape[:2]

        # Compute the mean prediction
        mean_preds = paddle.mean(preds, axis=0)
        mean_axis = tuple(range(0 if self.mean_over_samples else 1, mean_preds.ndim))
        # RMSE
        mse = self.mse(mean_preds, targets, mean_axis)

        # CRPS
        crps = self.crps(preds, targets)

        # SSR
        ssr = self.ssr(preds, targets, skill_metric=paddle.sqrt(mse), mean_axis=mean_axis)

        # compute negative log-likelihood
        # nll = self.nll(preds, targets, mean_axis=mean_axis)

        metric_dict = {"mse": mse, "ssr": ssr, "crps": crps}

        # MSE pre model
        if self.per_model:
            # next, compute the MSE for each model
            mse_per = self.mse(preds, paddle.expand_as(targets.unsqueeze(0), preds), False)
            mse_per_mean = paddle.mean(mse_per)
            metric_dict.update({"mse_per": mse_per, "mse_per_mean": mse_per_mean})
        return metric_dict
