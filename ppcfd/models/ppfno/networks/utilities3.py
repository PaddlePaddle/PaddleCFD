import sys
import paddle
import numpy as np
import scipy.io
import h5py
import operator
from functools import reduce
from functools import partial
import pynvml
device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    ).replace('cuda', 'gpu')


class UnitGaussianNormalizer(object):

    def __init__(self, x, eps=1e-05, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()
        self.mean = paddle.mean(x=x, axis=0)
        self.std = paddle.std(x=x, axis=0)
        self.eps = eps
        self.time_last = time_last

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps
                mean = self.mean[..., sample_idx]
        x = x * std + mean
        return x

    def to(self, device):
        if paddle.is_tensor(x=self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = paddle.to_tensor(data=self.mean).to(device)
            self.std = paddle.to_tensor(data=self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer(object):

    def __init__(self, x, eps=1e-05):
        super(GaussianNormalizer, self).__init__()
        self.mean = paddle.mean(x=x)
        self.std = paddle.std(x=x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = x * (self.std + self.eps) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda(blocking=True)
        self.std = self.std.cuda(blocking=True)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class RangeNormalizer(object):

    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = (paddle.min(x=x, axis=0), paddle.argmin(x=x, axis=0))[0].view(
            -1)
        mymax = (paddle.max(x=x, axis=0), paddle.argmax(x=x, axis=0))[0].view(
            -1)
        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = tuple(x.shape)
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = tuple(x.shape)
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(tuple(p.shape) + (2,) if p.
            is_complex() else tuple(p.shape)))
    return c


def paddle_memory_usage():
    cm = paddle.device.cuda.memory_allocated(
        ) / 1024 ** 3 if paddle.device.cuda.device_count() >= 1 else 0
    mm = paddle.device.cuda.max_memory_allocated(
        ) / 1024 ** 3 if paddle.device.cuda.device_count() >= 1 else 0
    rm = paddle.device.cuda.memory_reserved(
        ) / 1024 ** 3 if paddle.device.cuda.device_count() >= 1 else 0
    return f'{cm:.4g}GB/{mm:.4g}GB/{rm:.4g}GB (Current/MAX/Reserved'


def memory_usage(i=2):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    print(f'GPU {i}:')
    print(f'  Memory Total: {mem_info.total / 1024 ** 3:.2f} GB')
    print(f'  Memory Free: {mem_info.free / 1024 ** 3:.2f} GB')
    print(f'  Memory Used: {mem_info.used / 1024 ** 3:.2f} GB')
    print(f'  GPU Utilization: {util_info.gpu}%')
    print(f'  Memory Utilization: {util_info.memory}%')
    print(f'  Temperature: {temp} C')
    return None


def num_of_nans(x):
    return paddle.sum(x=paddle.isnan(x=x)).item()


def show_tensor_range(a):
    return f'{str(a)}:, {a}, max:, {paddle.max(x=a)}, min:, {paddle.min(x=a)}'
