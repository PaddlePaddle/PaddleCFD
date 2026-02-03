import enum

import numpy as np
import paddle


DEFAULT_W0 = 30.0


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = paddle.to_tensor(data=arr, dtype=paddle.float32)[timesteps]
    while len(tuple(res.shape)) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(shape=broadcast_shape)


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


def mean_flat(tensor):
    return paddle.mean(tensor, axis=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, paddle.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        (x if isinstance(x, paddle.Tensor) else paddle.to_tensor(x, dtype=tensor.dtype, place=tensor.place))
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0 + logvar2 - logvar1 + paddle.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )


class GaussianDiffusion:
    """
    Gaussian diffusion process for denoising diffusion probabilistic models (DDPM).

    Implements the forward diffusion process q(x_t|x_0) and reverse denoising process p(x_{t-1}|x_t).
    Supports various parameterizations (epsilon, x_0, x_{t-1}) and variance schedules.

    Reference:
        Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
        Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" (ICML 2021)

    Args:
        betas (np.ndarray): Noise schedule β_t for t=0,...,T-1.
        model_mean_type (ModelMeanType): Parameterization of model output.
        model_var_type (ModelVarType): Variance parameterization (fixed or learned).
        loss_type (LossType): Loss function type (MSE, KL, etc.).
        rescale_timesteps (bool, optional): Rescale timesteps to [0, 1000]. Defaults to False.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(tuple(betas.shape)) == 1, "betas must be 1-D"
        assert (betas > 0).astype("bool").all() and (betas <= 1).astype("bool").all()

        self.num_timesteps = int(tuple(betas.shape)[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert tuple(self.alphas_cumprod_prev.shape) == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_mean_variance(self, x_start, t):
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = paddle.randn(x_start.shape)

        sqrt_alpha_cumprod_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert tuple(x_t.shape) == tuple(xprev.shape)
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, tuple(x_t.shape)) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1,
                t,
                tuple(x_t.shape),
            )
            * x_t
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert tuple(x_t.shape) == tuple(eps.shape)
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape)) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape)) * eps
        )

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        B, C = tuple(x.shape)[:2]
        assert tuple(t.shape) == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert tuple(model_output.shape) == (B, C * 2, *tuple(x.shape)[2:])
            model_output, model_var_values = split(model_output, num_or_sections=C, axis=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = paddle.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, tuple(x.shape))
                max_log = _extract_into_tensor(np.log(self.betas), t, tuple(x.shape))
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = paddle.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, tuple(x.shape))
            model_log_variance = _extract_into_tensor(model_log_variance, t, tuple(x.shape))

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clip(min=-1, max=1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)
        assert tuple(model_mean.shape) == tuple(model_log_variance.shape) == tuple(pred_xstart.shape) == tuple(x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert tuple(x_start.shape) == tuple(x_t.shape)
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, tuple(x_t.shape)) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, tuple(x_t.shape)) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, tuple(x_t.shape))
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, tuple(x_t.shape))
        assert (
            tuple(posterior_mean.shape)[0]
            == tuple(posterior_variance.shape)[0]
            == tuple(posterior_log_variance_clipped.shape)[0]
            == tuple(x_start.shape)[0]
        )
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.astype(dtype="float32") * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var["mean"].astype(dtype="float32") + p_mean_var["variance"] * gradient.astype(
            dtype="float32"
        )
        return new_mean

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - paddle.sqrt(1 - alpha_bar) * cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        nonzero_mask = (t != 0).astype(dtype="float32").reshape([-1, *([1] * (len(tuple(x.shape)) - 1))])
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * paddle.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = paddle.randn(shape=shape)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        for i in indices:
            t = paddle.to_tensor(data=[i] * shape[0])
            with paddle.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, valid=False):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = paddle.randn(x_start.shape)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = split(model_output, C, axis=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = paddle.concat([model_output.detach(), model_var_values], axis=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=True,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            if valid is False:
                terms["mse"] = mean_flat((target - model_output) ** 2)
                if "vb" in terms:
                    terms["loss"] = terms["mse"] + terms["vb"]
                else:
                    terms["loss"] = terms["mse"]
            else:
                terms["valid_mse"] = mean_flat((target - model_output) ** 2)
                if "vb" in terms:
                    terms["loss"] = terms["valid_mse"] + terms["vb"]
                else:
                    terms["loss"] = terms["valid_mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = paddle.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = paddle.log(cdf_plus.clip(min=1e-12))
    log_one_minus_cdf_min = paddle.log((1.0 - cdf_min).clip(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = paddle.where(
        x < -0.999,
        log_cdf_plus,
        paddle.where(x > 0.999, log_one_minus_cdf_min, paddle.log(cdf_delta.clip(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + paddle.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * paddle.pow(x, 3))))


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class SpacedDiffusion(GaussianDiffusion):
    """
    Accelerated diffusion process that skips timesteps for faster sampling.

    Implements DDIM-style sampling by using a subset of timesteps from the original
    diffusion process, enabling faster inference without retraining the model.

    Reference:
        Song et al. "Denoising Diffusion Implicit Models" (ICLR 2021)

    Args:
        use_timesteps (Sequence[int]): Collection of timesteps to retain from original process
                                       (e.g., [0, 10, 20, ..., 1000] for 100-step sampling).
        **kwargs: Additional arguments for base GaussianDiffusion (betas, model_mean_type, etc.).
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps, self.original_num_steps)

    def _scale_timesteps(self, t):
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = paddle.to_tensor(data=self.timestep_map, dtype=ts.dtype)  # , place=ts.place
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.astype(dtype="float32") * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
