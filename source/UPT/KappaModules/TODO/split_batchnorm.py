import torch
import torch.nn.functional as F
# noinspection PyProtectedMember
from torch.nn.modules.batchnorm import _BatchNorm


class SplitBatchNorm(_BatchNorm):
    def __init__(self, *args, batch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    """
    adapted from torch.nn.SyncBatchNorm.convert_sync_batchnorm
    seperates the input tensor into chunks of batch_size
    used in contrastive learning to avoid normalization statistics to be calculation over multiple views
    """

    def _check_input_dim(self, x):
        if x.dim() < 2:
            raise ValueError("expected at least 2D input (got {}D input)".format(x.dim()))

    @staticmethod
    def _check_non_zero_input_channels(x):
        if x.size(1) == 0:
            raise ValueError("SplitBatchNorm number of input channels should be non-zero")

    def forward(self, x):
        self._check_input_dim(x)
        self._check_non_zero_input_channels(x)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None

        # Don't split batchnorm stats in inference mode (model.eval()).
        if bn_training and self.training:
            if torch.is_tensor(x):
                # chunk if input is tensor
                was_tensor = True
                assert len(x) % self.batch_size == 0
                x = x.chunk(len(x) // self.batch_size)
            else:
                # if not tensor -> expecting chunks
                was_tensor = False
                assert isinstance(x, (list, tuple)) and all(len(xx) == self.batch_size for xx in x)

            # split forward
            results = [
                F.batch_norm(
                    chunk,
                    running_mean,
                    running_var,
                    self.weight,
                    self.bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
                for chunk in x
            ]

            # concat if input was tensor
            if was_tensor:
                results = torch.concat(results)

            return results
        else:
            return F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )

    @classmethod
    def convert_split_batchnorm(cls, module, batch_size):
        module_output = module
        if isinstance(module, _BatchNorm):
            module_output = SplitBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                batch_size=batch_size,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_split_batchnorm(child, batch_size=batch_size))
        del module
        return module_output
