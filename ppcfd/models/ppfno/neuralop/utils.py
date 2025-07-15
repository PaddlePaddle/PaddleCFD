import sys
sys.path.append('/home/paddle_project')
import paddle
import os
from typing import List, Optional, Union
from math import prod
# import wandb
import warnings


class UnitGaussianNormalizer:

    def __init__(self, x, eps=1e-05, reduce_dim=[0], verbose=True):
        super().__init__()
        msg = (
            'neuralop.utils.UnitGaussianNormalizer has been deprecated. Please use the newer neuralop.datasets.UnitGaussianNormalizer instead.'
            )
        warnings.warn(msg, DeprecationWarning)
        n_samples, *shape = tuple(x.shape)
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim
        
        self.mean = paddle.mean(x=x, axis=reduce_dim, keepdim=True).squeeze(
            axis=0)
        self.std = paddle.std(x=x, axis=reduce_dim, keepdim=True).squeeze(axis
            =0)
        self.eps = eps
        if verbose:
            print(
                f'UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}.'
                )
            print(
                f'   Mean and std of shape {tuple(self.mean.shape)}, eps={eps}'
                )

    def encode(self, x):
        x -= self.mean
        x /= self.std + self.eps
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(tuple(self.mean.shape)) == len(tuple(sample_idx[0].shape)):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if len(tuple(self.mean.shape)) > len(tuple(sample_idx[0].shape)):
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        x *= std
        x += mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def count_model_params(model):
    """Returns the total number of parameters of a Paddle model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum([(p.size * 2 if p.is_complex() else p.size) for p in model.
        parameters()])


def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor paddle.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tuple(tensor.shape))
    else:
        dims = [tuple(tensor.shape)[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2 * n_params
    return n_params


# def wandb_login(api_key_file='../config/wandb_api_key.txt', key=None):
    # if key is None:
        # key = get_wandb_api_key(api_key_file)
    # wandb.login(key=key)


def set_wandb_api_key(api_key_file='../config/wandb_api_key.txt'):
    try:
        os.environ['WANDB_API_KEY']
    except KeyError:
        with open(api_key_file, 'r') as f:
            key = f.read()
        os.environ['WANDB_API_KEY'] = key.strip()


def get_wandb_api_key(api_key_file='../config/wandb_api_key.txt'):
    try:
        return os.environ['WANDB_API_KEY']
    except KeyError:
        with open(api_key_file, 'r') as f:
            key = f.read()
        return key.strip()


def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.

    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
    """
    T = tuple(signal.shape)[0]
    signal = signal.view(T, n_observations, n_observations)
    if normalize:
        signal = paddle.fft.fft2(x=signal)
    else:
        signal = paddle.fft.rfft2(signal, s=(n_observations, n_observations),
            normalized=False)
    k_max = n_observations // 2
    wavenumers = paddle.concat(x=(paddle.arange(start=0, end=k_max, step=1),
        paddle.arange(start=-k_max, end=0, step=1)), axis=0).tile(repeat_times
        =[n_observations, 1])
    k_x = wavenumers.transpose(perm=utils.dim2perm(wavenumers.ndim, 0, 1))
    k_y = wavenumers
    sum_k = paddle.abs(x=k_x) + paddle.abs(x=k_y)
    sum_k = sum_k
    index = -1.0 * paddle.ones(shape=(n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]
    spectrum = paddle.zeros(shape=(T, n_observations))
    for j in range(1, n_observations + 1):
        ind = paddle.where(index == j)
        spectrum[:, j - 1] = signal[:, ind[0], ind[1]].sum(axis=1).abs() ** 2
    spectrum = spectrum.mean(axis=0)
    return spectrum


Number = Union[float, int]


def validate_scaling_factor(scaling_factor: Union[None, Number, List[Number
    ], List[List[Number]]], n_dim: int, n_layers: Optional[int]=None) ->Union[
    None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim
        return [[float(scaling_factor)] * n_dim] * n_layers
    if isinstance(scaling_factor, list) and len(scaling_factor) > 0 and all([
        isinstance(s, (float, int)) for s in scaling_factor]):
        return [([float(s)] * n_dim) for s in scaling_factor]
    if isinstance(scaling_factor, list) and len(scaling_factor) > 0 and all([
        isinstance(s, (float, int)) for s in scaling_factor]):
        return [([float(s)] * n_dim) for s in scaling_factor]
    if isinstance(scaling_factor, list) and len(scaling_factor) > 0 and all([
        isinstance(s, list) for s in scaling_factor]):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor
    return None
