import paddle
from typing import List, Union
import sys
# sys.path.append('/home/src')
from ..utils import validate_scaling_factor


class DomainPadding(paddle.nn.Layer):
    """Applies domain padding scaled automatically to the input's resolution

    Parameters
    ----------
    domain_padding : float or list
        typically, between zero and one, percentage of padding to use
        if a list, make sure if matches the dim of (d1, ..., dN)
    padding_mode : {'symmetric', 'one-sided'}, optional
        whether to pad on both sides, by default 'one-sided'
    output_scaling_factor : int ; default is 1

    Notes
    -----
    This class works for any input resolution, as long as it is in the form
    `(batch-size, channels, d1, ...., dN)`
    """

    def __init__(self, domain_padding, padding_mode='one-sided',
        output_scaling_factor: Union[int, List[int]]=1):
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode.lower()
        if output_scaling_factor is None:
            output_scaling_factor = 1
        self.output_scaling_factor: Union[int, List[int]
            ] = output_scaling_factor
        self._padding = dict()
        self._unpad_indices = dict()

    def forward(self, x):
        """forward pass: pad the input"""
        self.pad(x)

    def pad(self, x, verbose=False):
        """Take an input and pad it by the desired fraction

        The amount of padding will be automatically scaled with the resolution
        """
        resolution = tuple(x.shape)[2:]
        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)] * len(resolution
                )
        assert len(self.domain_padding) == len(resolution
            ), 'domain_padding length must match the number of spatial/time dimensions (excluding batch, ch)'
        output_scaling_factor = self.output_scaling_factor
        if not isinstance(self.output_scaling_factor, list):
            output_scaling_factor: List[float] = validate_scaling_factor(self
                .output_scaling_factor, len(resolution), n_layers=None)
        try:
            padding = self._padding[f'{resolution}']
            return paddle.nn.functional.pad(x=x, pad=padding, mode=
                'constant', pad_from_left_axis=False)
        except KeyError:
            padding = [round(p * r) for p, r in zip(self.domain_padding,
                resolution)]
            if verbose:
                print(
                    f'Padding inputs of resolution={resolution} with padding={padding}, {self.padding_mode}'
                    )
            output_pad = padding
            output_pad = [round(i * j) for i, j in zip(
                output_scaling_factor, output_pad)]
            padding = padding[::-1]
            if self.padding_mode == 'symmetric':
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_end = None
                        padding_start = None
                    else:
                        padding_end = p
                        padding_start = -p
                    unpad_list.append(slice(padding_end, padding_start, None))
                unpad_indices = (Ellipsis,) + tuple(unpad_list)
                padding = [i for p in padding for i in (p, p)]
            elif self.padding_mode == 'one-sided':
                unpad_list = list()
                for p in output_pad:
                    if p == 0:
                        padding_start = None
                    else:
                        padding_start = -p
                    unpad_list.append(slice(None, padding_start, None))
                unpad_indices = (Ellipsis,) + tuple(unpad_list)
                padding = [i for p in padding for i in (0, p)]
            else:
                raise ValueError(f'Got padding_mode={self.padding_mode}')
            self._padding[f'{resolution}'] = padding
            padded = paddle.nn.functional.pad(x=x, pad=padding, mode=
                'constant', pad_from_left_axis=False)
            output_shape = tuple(padded.shape)[2:]
            output_shape = [round(i * j) for i, j in zip(
                output_scaling_factor, output_shape)]
            self._unpad_indices[f'{[i for i in output_shape]}'] = unpad_indices
            return padded

    def unpad(self, x):
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices[f'{list(tuple(x.shape)[2:])}']
        return x[unpad_indices]
