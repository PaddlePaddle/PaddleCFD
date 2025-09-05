from math import sqrt

import paddle
from einops import rearrange
from einops import reduce
from tqdm import tqdm


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def log(t, eps=1e-20):
    return paddle.log(x=t.clip(min=eps))


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class ElucidatedDiffusion(paddle.nn.Layer):
    def __init__(
        self,
        net,
        *,
        image_size_h,
        image_size_w,
        channels=3,
        num_sample_steps=32,
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
        S_churn=80,
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
    ):
        super().__init__()
        self.self_condition = net.self_condition
        self.net = net
        self.channels = channels
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).place

    def c_skip(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    def preconditioned_network_forward(self, noised_images, sigma, self_cond=None, clamp=False):
        batch = noised_images.shape[0]
        if isinstance(sigma, float):
            sigma = paddle.full(shape=(batch,), fill_value=sigma)
        padded_sigma = rearrange(sigma, "b -> b 1 1 1")
        net_out = self.net(self.c_in(padded_sigma) * noised_images, self.c_noise(sigma), self_cond)
        out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out
        if clamp:
            out = out.clip(min=-1.0, max=1.0)
        return out

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        N = num_sample_steps
        inv_rho = 1 / self.rho
        steps = paddle.arange(dtype="float32", end=num_sample_steps)
        sigmas = (
            self.sigma_max**inv_rho + steps / (N - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho
        sigmas = paddle.nn.functional.pad(x=sigmas, pad=(0, 1), value=0.0, pad_from_left_axis=False)
        return sigmas

    @paddle.no_grad()
    def sample(self, self_cond, batch_size=None, num_sample_steps=None, clamp=True):
        batch_size = tuple(self_cond.shape)[0]
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        shape = batch_size, self.channels, self.image_size_h, self.image_size_w
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = paddle.where(
            condition=(sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            x=min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            y=0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
        init_sigma = sigmas[0]
        images = init_sigma * paddle.randn(shape=shape)
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc="sampling time step"):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))
            eps = self.S_noise * paddle.randn(shape=shape)
            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat**2 - sigma**2) * eps
            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp=clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat
            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma
            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    images_next, sigma_next, self_cond, clamp=clamp
                )
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )
            images = images_next
        images = images.clip(min=-1.0, max=1.0)
        return unnormalize_to_zero_to_one(images)

    @paddle.no_grad()
    def sample_using_dpmpp(self, self_cond, batch_size=None, num_sample_steps=None):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """
        batch_size = tuple(self_cond.shape)[0]
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)
        sigmas = self.sample_schedule(num_sample_steps)
        shape = batch_size, self.channels, self.image_size_h, self.image_size_w
        images = sigmas[0] * paddle.randn(shape=shape)
        sigma_fn = lambda t: t.neg().exp()  # noqa
        t_fn = lambda sigma: sigma.log().neg()  # noqa
        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(images, sigmas[i].item())
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = -1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised
            images = sigma_fn(t_next) / sigma_fn(t) * images - (-h).expm1() * denoised_d
            old_denoised = denoised
        images = images.clip(min=-1.0, max=1.0)
        return unnormalize_to_zero_to_one(images)

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * paddle.randn(shape=(batch_size,))).exp()

    def forward(self, images, self_cond=None):
        (batch_size, c, h, w, image_size_h, image_size_w, channels) = (
            *tuple(images.shape),
            self.image_size_h,
            self.image_size_w,
            self.channels,
        )
        assert (
            h == image_size_h and w == image_size_w
        ), f"height and width of image must be {image_size_h}, {image_size_w}"
        assert c == channels, "mismatch of image channels"
        images = normalize_to_neg_one_to_one(images)
        sigmas = self.noise_distribution(batch_size)

        padded_sigmas = rearrange(sigmas, "b -> b 1 1 1")
        noise = paddle.randn(shape=images.shape, dtype=images.dtype)
        noised_images = images + padded_sigmas * noise

        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)
        losses = paddle.nn.functional.mse_loss(input=denoised, label=images, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        return losses.mean()
