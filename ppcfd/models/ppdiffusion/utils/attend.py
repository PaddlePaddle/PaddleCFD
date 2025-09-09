from collections import namedtuple
from functools import wraps

import paddle
import paddle.nn.functional as F

from .utils import custom_sdp_kernel


AttentionConfig = namedtuple("AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


class Attend(paddle.nn.Layer):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = paddle.nn.Dropout(p=dropout)
        self.flash = flash
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not paddle.is_compiled_with_cuda() or not flash:
            return

        # device_properties = paddle.device.cuda.get_device_properties()
        # device_version = version.parse(device_properties.compute_capability)
        # if device_version > version.parse('8.0'):
        #     print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
        #     self.cuda_config = AttentionConfig(True, False, False)
        # else:
        if True:
            print_once("Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda")
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, _, _, _, _, is_cuda = (  # heads/q_len/k_len unused
            *tuple(q.shape),
            tuple(k.shape)[-2],
            q.place.is_gpu_place(),
        )
        if exists(self.scale):
            default_scale = tuple(q.shape)[-1]
            q = q * (self.scale / default_scale)
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        config = self.cuda_config if is_cuda else self.cpu_config

        with custom_sdp_kernel(
            enable_math=config.enable_math,
            enable_flash=config.enable_flash,
            enable_mem_efficient=config.enable_mem_efficient,
        ):
            q = q.transpose([0, 2, 1, 3])
            k = k.transpose([0, 2, 1, 3])
            v = v.transpose([0, 2, 1, 3])
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0).transpose(
                [0, 2, 1, 3]
            )
        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        # q_len, k_len = tuple(q.shape)[-2], tuple(k.shape)[-2]   # unused
        if self.flash:
            return self.flash_attn(q, k, v)
        scale = default(self.scale, tuple(q.shape)[-1] ** -0.5)
        sim = paddle.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = F.softmax(sim, axis=-1)
        attn = self.attn_dropout(attn)
        out = paddle.einsum("b h i j, b h j d -> b h i d", attn, v)
        return out
