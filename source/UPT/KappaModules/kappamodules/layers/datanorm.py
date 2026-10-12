import einops
import paddle


class DataNormStateDictPreHook:
    def __call__(self, module, *args, **kwargs):
        if paddle.distributed.is_initialized():
            module.finish()


class DataNorm(paddle.nn.Layer):
    def __init__(
        self, dim, eps=1e-06, channel_first=True, gather_mode="global", frozen=False
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.frozen = frozen
        self.channel_first = channel_first
        assert gather_mode in ["global", "none"]
        self.gather_mode = gather_mode
        self.register_buffer("mean", paddle.zeros(dim))
        self.register_buffer("var", paddle.ones(dim))
        self.register_buffer("num_batches_tracked", paddle.tensor(0.0))
        self.mean_buffer = []
        self.var_buffer = []
        self._async_handle = None
        self.register_state_dict_pre_hook(DataNormStateDictPreHook())

    def _x_to_stats(self, x):
        if self.channel_first:
            x = einops.rearrange(x, "bs dim ... -> (bs ...) dim")
        else:
            x = einops.rearrange(x, "bs ... dim -> (bs ...) dim")
        mean = x.mean(dim=0)
        var = x.var(axis=0, unbiased=False)
        return mean, var

    def finish(self):
        self._update_stats(inplace=True)

    def _update_stats(self, inplace):
        if self._async_handle is not None:
            self._async_handle.wait()
            x = self._async_handle.get_future().value()
            x = paddle.concat(x)
            self._async_handle = None
            xmean, xvar = self._x_to_stats(x)
            self.mean_buffer.append(xmean)
            self.var_buffer.append(xvar)
        if len(self.mean_buffer) == 0:
            assert len(self.var_buffer) == 0
            return
        assert len(self.mean_buffer) == 1, len(self.mean_buffer)
        assert len(self.var_buffer) == 1, len(self.var_buffer)
        mean = self.mean_buffer[0]
        var = self.var_buffer[0]
        self.mean_buffer.clear()
        self.var_buffer.clear()
        self.num_batches_tracked += 1
        momentum = 1 / self.num_batches_tracked
        if mean.shape != self.mean.shape:
            raise NotImplementedError(
                f"invalid stat shapes -> most likely due to GPUs having different batch_sizes -> set gather_mode='none'"
            )
        if inplace:
            self.mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            self.var.mul_(1 - momentum).add_(var, alpha=momentum)
        else:
            self.mean = (self.mean * (1 - momentum)).add_(mean, alpha=momentum)
            self.var = (self.var * (1 - momentum)).add_(var, alpha=momentum)

    def forward(self, x):
        if self.training:
            if len(x) == 1:
                raise NotImplementedError(
                    "DataNorm batch_size=1 requires syncing features instead of stats"
                )
        if self.gather_mode == "global":
            if paddle.distributed.is_initialized():
                if self._async_handle is not None:
                    self._update_stats(inplace=True)
                if self.training and not self.frozen:
                    assert self._async_handle is None
                    tensor_list = [
                        paddle.zeros_like(x)
                        for _ in range(paddle.distributed.get_world_size())
                    ]
                    self._async_handle = paddle.distributed.all_gather(
                        tensor_list=tensor_list, tensor=x, sync_op=not True
                    )
        elif self.gather_mode == "none":
            pass
        else:
            raise NotImplementedError
        og_x = x
        if not self.channel_first:
            x = einops.rearrange(x, "bs ... dim -> bs dim ...")
        x = paddle.nn.functional.batch_norm(
            x=x,
            running_mean=self.mean,
            running_var=self.var,
            epsilon=self.eps,
            training=False,
        )
        if not self.channel_first:
            x = einops.rearrange(x, "bs dim ... -> bs ... dim")
        if self.training and not self.frozen and self._async_handle is None:
            with paddle.no_grad():
                xmean, xvar = self._x_to_stats(og_x)
                self.mean_buffer.append(xmean)
                self.var_buffer.append(xvar)
            self._update_stats(inplace=not x.requires_grad)
        return x
