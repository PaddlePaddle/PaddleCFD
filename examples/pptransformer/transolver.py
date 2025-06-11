# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code is heavily based on paper "Transolver: A Fast Transformer Solver for PDEs on General Geometries",
# we use paddle to reproduce the results of the paper

import random
import sys
from multiprocessing import Value
from typing import Optional

import paddle
from einops import rearrange
from einops import repeat
from paddle._typing import DTypeLike
from paddle.nn import Dropout
from paddle.nn import Linear

sys.path.append("/workspace/DNNFluid_Car/DNNFluid-Car/")
from ppcfd.networks.KAN import KAN
from ppcfd.neuralop.models import FNO

ACTIVATION = {
    "gelu": paddle.nn.GELU,
    "tanh": paddle.nn.Tanh,
    "sigmoid": paddle.nn.Sigmoid,
    "relu": paddle.nn.ReLU,
    "leaky_relu": paddle.nn.LeakyReLU(negative_slope=0.1),
    "softplus": paddle.nn.Softplus,
    "ELU": paddle.nn.ELU,
    "silu": paddle.nn.Silu,
}


def print_gpu_memory(id, card_list):
    import paddle.distributed as dist

    if dist.get_world_size() > 1:
        card1 = card_list[0]
        card2 = card_list[1]
        card3 = card_list[2]
        card4 = card_list[3]
        print(f"index:{id}")
        print(
            f"多卡:代码执行到此处时，消耗的GPU:{card1}显存:{paddle.device.cuda.max_memory_allocated(card1) / (1024 ** 3):.2f} GB"
        )
        print(
            f"多卡:代码执行到此处时，消耗的GPU:{card2}显存:{paddle.device.cuda.max_memory_allocated(card2) / (1024 ** 3):.2f} GB"
        )
        print(
            f"多卡:代码执行到此处时，消耗的GPU:{card3}显存:{paddle.device.cuda.max_memory_allocated(card3) / (1024 ** 3):.2f} GB"
        )
        print(
            f"多卡:代码执行到此处时，消耗的GPU:{card4}显存:{paddle.device.cuda.max_memory_allocated(card4) / (1024 ** 3):.2f} GB"
        )
        print(f"多卡:显存日志保存在 ./log/worklog.{card1}")
        print(f"多卡:显存日志保存在 ./log/worklog.{card2}")
        print(f"多卡:显存日志保存在 ./log/worklog.{card3}")
        print(f"多卡:显存日志保存在 ./log/worklog.{card4}")
    else:
        pass
        print(f"index:{id}")
        print(
            f"单卡:代码执行到此处时，消耗的MAX GPU显存:{paddle.device.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB"
        )
        print(
            f"单卡:代码执行到此处时，消耗的CURRENT GPU显存:{paddle.device.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
        )


class Physics_Attention_1D(paddle.nn.Layer):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, attn_type=""
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        out_0 = paddle.create_parameter(
            shape=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).shape,
            dtype=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.ones(shape=[1, heads, 1, 1]) * 0.5
            ),
        )
        out_0.stop_gradient = not True
        self.temperature = out_0
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_slice = paddle.nn.Linear(
            in_features=dim_head, out_features=slice_num
        )
        for la in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(la.weight)
        self.to_q = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_k = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_v = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=dim),
            paddle.nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x = B, N, C
        B, N, C = tuple(x.shape)

        # self.in_project_fx(x) = B, N, n_hidden
        # fx_mid = B, heads, N, dim_head
        fx_mid = (
            self.in_project_fx(x)
            .reshape((B, N, self.heads, self.dim_head))
            .transpose(perm=[0, 2, 1, 3])
        )

        # self.in_project_fx(x) = B, N, n_hidden
        # x_mid = B, heads, N, dim_head
        x_mid = (
            self.in_project_x(x)
            .reshape((B, N, self.heads, self.dim_head))
            .transpose(perm=[0, 2, 1, 3])
        )

        # self.in_project_slice(x_mid) = B, heads, N, slice_num
        # self.temperature = 1, heads, 1, 1
        # self.slice_weights = B, heads, N, slice_num
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.matmul(x=slice_weights, y=fx_mid, transpose_x=True)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            (1, 1, 1, self.dim_head)
        )

        # slice_token = slice_num, slice_num
        if not self.training:
            del (x, fx_mid, x_mid, slice_norm)
            paddle.device.cuda.empty_cache()

        # qkv_slice_token = B , N , slice_num, slice_num
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        # Attention
        x = k_slice_token

        # dots: B, heads, slice_num, slice_num
        # attn: B, heads, slice_num, slice_num
        dots = paddle.matmul(x=q_slice_token, y=x, transpose_y=True) * self.scale
        attn = self.softmax(dots)

        attn = self.dropout(attn)
        # attn: B, heads, slice_num, slice_num
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)

        out_x = paddle.matmul(x=slice_weights, y=out_slice_token)
        out_x = paddle.transpose(
            out_x, perm=[0, 2, 1, 3]
        )  # Now out_x.shape = [b, n, h, d]
        out_x = paddle.reshape(
            out_x, shape=[out_x.shape[0], out_x.shape[1], -1]
        )  # Now out_x.shape = [b, n, h*d]
        if not self.training:
            del (
                x,
                slice_weights,
                slice_token,
                q_slice_token,
                k_slice_token,
                v_slice_token,
                dots,
                attn,
                out_slice_token,
            )
            paddle.device.cuda.empty_cache()
        out_x = self.to_out(out_x)
        if not self.training:
            paddle.device.cuda.empty_cache()
        return out_x


class Eidetic_States_Attention(paddle.nn.Layer):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, attn_type=""
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.distribution = paddle.distribution.Uniform(low=0.0, high=1.0)

        # New
        self.temperature_0 = paddle.create_parameter(
            shape=[1, heads, 1, 1],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.ones(shape=[1, heads, 1, 1]) * 0.5
            ),
        )
        self.temperature_0.stop_gradient = False
        self.temperature = paddle.nn.Linear(
            in_features=inner_dim, out_features=1
        )  # +out_0
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_slice = paddle.nn.Linear(
            in_features=dim_head, out_features=slice_num
        )
        for la in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(la.weight)
        self.to_q = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_k = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_v = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=dim),
            paddle.nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # x = B, N, C
        B, N, C = tuple(x.shape)

        # self.in_project_fx(x) = B, N, n_hidden
        # fx_mid = B, heads, N, dim_head
        # fx_mid = (
        #     self.in_project_fx(x).reshape((B, N, self.heads, self.dim_head)).transpose(perm=[0, 2, 1, 3])
        # )

        # self.in_project_fx(x) = B, N, n_hidden
        # x_mid = B, heads, N, dim_head
        x_mid = (
            self.in_project_x(x)
            .reshape((B, N, self.heads, self.dim_head))
            .transpose(perm=[0, 2, 1, 3])
        )

        # self.in_project_slice(x_mid) = B, heads, N, slice_num
        # self.temperature = 1, heads, 1, 1
        # self.slice_weights = B, heads, N, slice_num
        ada_temp = self.temperature_0 + self.temperature(x_mid)
        epsilon = self.distribution.sample(ada_temp.shape)
        slice_weights = (
            self.in_project_slice(x_mid) + paddle.log(-paddle.log(epsilon))
        ) / ada_temp
        slice_weights = paddle.nn.functional.softmax(slice_weights)
        # slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.matmul(x=slice_weights, y=x_mid, transpose_x=True)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            (1, 1, 1, self.dim_head)
        )

        # slice_token = slice_num, slice_num
        if not self.training:
            del (x, x_mid, slice_norm)
            paddle.device.cuda.empty_cache()

        # qkv_slice_token = B , N , slice_num, slice_num
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        # Attention
        x = k_slice_token

        # dots: B, heads, slice_num, slice_num
        # attn: B, heads, slice_num, slice_num
        dots = paddle.matmul(x=q_slice_token, y=x, transpose_y=True) * self.scale
        attn = self.softmax(dots)

        attn = self.dropout(attn)
        # attn: B, heads, slice_num, slice_num
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)

        out_x = paddle.matmul(x=slice_weights, y=out_slice_token)
        out_x = paddle.transpose(
            out_x, perm=[0, 2, 1, 3]
        )  # Now out_x.shape = [b, n, h, d]
        out_x = paddle.reshape(
            out_x, shape=[out_x.shape[0], out_x.shape[1], -1]
        )  # Now out_x.shape = [b, n, h*d]
        if not self.training:
            del (
                x,
                slice_weights,
                slice_token,
                q_slice_token,
                k_slice_token,
                v_slice_token,
                dots,
                attn,
                out_slice_token,
            )
            paddle.device.cuda.empty_cache()
        out_x = self.to_out(out_x)
        if not self.training:
            paddle.device.cuda.empty_cache()
        return out_x


class FNO_Slice(paddle.nn.Layer):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, attn_type="l1"
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.dropout = paddle.nn.Dropout(p=dropout)
        out_0 = paddle.create_parameter(
            shape=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).shape,
            dtype=(paddle.ones(shape=[1, heads, 1, 1]) * 0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.ones(shape=[1, heads, 1, 1]) * 0.5
            ),
        )
        out_0.stop_gradient = not True
        self.temperature = out_0
        self.in_project_x = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_fx = paddle.nn.Linear(in_features=dim, out_features=inner_dim)
        self.in_project_slice = paddle.nn.Linear(
            in_features=dim_head, out_features=slice_num
        )
        for la in [self.in_project_slice]:
            init_Orthogonal = paddle.nn.initializer.Orthogonal()
            init_Orthogonal(la.weight)
        self.to_q = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_k = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_v = paddle.nn.Linear(
            in_features=dim_head, out_features=dim_head, bias_attr=False
        )
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=dim),
            paddle.nn.Dropout(p=dropout),
        )

        self.temperature_fno = FNO(
            (8, 8),
            in_channels=8,
            hidden_channels=8,
            out_channels=8,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=0.125,
            factorization="tucker",
            norm="group_norm",
            rank=0.4,
        )

    def forward(self, x):
        # x = B, N, C
        B, N, C = tuple(x.shape)

        # self.in_project_fx(x) = B, N, n_hidden
        # fx_mid = B, heads, N, dim_head
        fx_mid = (
            self.in_project_fx(x)
            .reshape((B, N, self.heads, self.dim_head))
            .transpose(perm=[0, 2, 1, 3])
        )

        # self.in_project_fx(x) = B, N, n_hidden
        # x_mid = B, heads, N, dim_head
        x_mid = (
            self.in_project_x(x)
            .reshape((B, N, self.heads, self.dim_head))
            .transpose(perm=[0, 2, 1, 3])
        )

        # self.in_project_slice(x_mid) = B, heads, N, slice_num
        # self.temperature = 1, heads, 1, 1
        # self.slice_weights = B, heads, N, slice_num
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(axis=2)
        slice_token = paddle.matmul(x=slice_weights, y=fx_mid, transpose_x=True)
        slice_token = slice_token / (slice_norm + 1e-05)[:, :, :, None].tile(
            (1, 1, 1, self.dim_head)
        )
        slice_token = self.temperature_fno(slice_token)

        # slice_token = slice_num, slice_num
        if not self.training:
            del (x, fx_mid, x_mid, slice_norm)
            paddle.device.cuda.empty_cache()

        # qkv_slice_token = B , N , slice_num, slice_num
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        # Attention
        x = k_slice_token

        # dots: B, heads, slice_num, slice_num
        # attn: B, heads, slice_num, slice_num
        dots = paddle.matmul(x=q_slice_token, y=x, transpose_y=True) * self.scale
        attn = self.softmax(dots)

        attn = self.dropout(attn)
        # attn: B, heads, slice_num, slice_num
        out_slice_token = paddle.matmul(x=attn, y=v_slice_token)

        out_x = paddle.matmul(x=slice_weights, y=out_slice_token)
        out_x = paddle.transpose(
            out_x, perm=[0, 2, 1, 3]
        )  # Now out_x.shape = [b, n, h, d]
        out_x = paddle.reshape(
            out_x, shape=[out_x.shape[0], out_x.shape[1], -1]
        )  # Now out_x.shape = [b, n, h*d]
        if not self.training:
            del (
                x,
                slice_weights,
                slice_token,
                q_slice_token,
                k_slice_token,
                v_slice_token,
                dots,
                attn,
                out_slice_token,
            )
            paddle.device.cuda.empty_cache()
        out_x = self.to_out(out_x)
        if not self.training:
            paddle.device.cuda.empty_cache()
        return out_x


class Linear_Attention(paddle.nn.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self,
        hidden_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        attn_type="linear",
        **kwargs,
    ):
        super(Linear_Attention, self).__init__()
        assert hidden_dim % heads == 0
        # key, query, value projections for all heads
        self.key = Linear(hidden_dim, hidden_dim)
        self.query = Linear(hidden_dim, hidden_dim)
        self.value = Linear(hidden_dim, hidden_dim)
        # regularization
        self.attn_drop = Dropout(dropout)
        # output projection
        self.proj = Linear(hidden_dim, hidden_dim)
        self.n_head = heads
        self.attn_type = attn_type
        self.softmax = paddle.nn.Softmax(axis=-1)

    """
        Linear Attention and Linear Cross Attention (if y is provided)
        -----------------------------------------------------------------------------------
        类型         归一化方法	      缩放因子D_inv 计算方式	            特点
        -----------------------------------------------------------------------------------
        l1          Softmax	        动态计算（基于输入数据的累积和）	    经典注意力变体，动态缩放
        galerkin	Softmax	        固定参数 1/T2	                    理论驱动，预定义归一化
        l2          L1 范数归一化	  动态计算（含绝对值操作）	            稀疏性增强，符号保留
        -----------------------------------------------------------------------------------
    """

    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.shape
        _, T2, _ = y.shape
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = (
            self.query(x)
            .reshape([B, T1, self.n_head, C // self.n_head])
            .transpose([0, 2, 1, 3])
        )  # (B, nh, T, hs)

        k = (
            self.key(y)
            .reshape([B, T2, self.n_head, C // self.n_head])
            .transpose([0, 2, 1, 3])
        )  # (B, nh, T, hs)
        v = (
            self.value(y)
            .reshape([B, T2, self.n_head, C // self.n_head])
            .transpose([0, 2, 1, 3])
        )  # (B, nh, T, hs)

        if self.attn_type == "l1":
            q = self.softmax(q)
            k = self.softmax(k)
            k_cumsum = k.sum(axis=-2, keepdim=True)
            D_inv = 1.0 / (q * k_cumsum).sum(axis=-1, keepdim=True)  # normalized
        elif self.attn_type == "galerkin":
            q = self.softmax(q)
            k = self.softmax(k)
            D_inv = 1.0 / T2  # galerkin
        elif self.attn_type == "l2":  # still use l1 normalization
            q = q / q.norm(axis=-1, keepdim=True, p=1)
            k = k / k.norm(axis=-1, keepdim=True, p=1)
            k_cumsum = k.sum(axis=-2, keepdim=True)
            D_inv = 1.0 / (q * k_cumsum).abs().sum(axis=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        # torch.Size([4, 1, 10086, 128]) torch.Size([4, 1, 10086, 128])
        context = k.transpose([0, 1, 3, 2]) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, "b h n d -> b n (h d)")
        y = self.proj(y)
        return y


class MLP(paddle.nn.Layer):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=n_input, out_features=n_hidden), act()
        )
        self.linear_post = paddle.nn.Linear(in_features=n_hidden, out_features=n_output)
        self.linears = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Sequential(
                    paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden), act()
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


ATTENTION = {
    "Physics_Attention_1D": Physics_Attention_1D,
    "Eidetic_States_Attention": Eidetic_States_Attention,
    "FNO_Slice": FNO_Slice,
    "linearAtten": Linear_Attention,
}


class Transolver_block(paddle.nn.Layer):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        attn="FNO_Slice",
        attn_type="l1",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        _Attention_Layer = ATTENTION[attn]
        self.Attn = _Attention_Layer(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            attn_type=attn_type,
        )
        self.ln_2 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        self.pp_layer = paddle.nn.Identity()
        if self.last_layer:
            self.ln_3 = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
            self.mlp2 = paddle.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, fx):
        # residual connection
        fx = self.Attn(self.ln_1(fx)) + fx
        # residual connection
        fx = self.mlp(self.ln_2(fx)) + fx
        fx = self.pp_layer(fx)
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Reynolds_Embedding(paddle.nn.Layer):
    def __init__(self, ref=8):
        super(Reynolds_Embedding, self).__init__()
        self.ref = ref
        self.learnable_similarity = paddle.nn.Linear(2, 1)
        self.scaling = [0.8, 1.2]
        self.density = 1.0
        self.in_vel = 30.0
        self.mu = 1.0

    def forward(self, x, in_vel):
        scaler = random.choice(self.scaling)
        characterisctic_length = x[0].max() - x[0].min()
        reynolds_number = self.density * in_vel * characterisctic_length / self.mu
        x = x / scaler
        in_vel = paddle.concat([in_vel, reynolds_number])
        in_vel = self.learnable_similarity(in_vel) * scaler
        return x, in_vel


class Transolver_0513_0(paddle.nn.Layer):
    def __init__(
        self,
        space_dim=3,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        act="gelu",
        attn="FNO_Slice",
        attn_type="linear",
        mlp_ratio=1,
        fun_dim=0,
        out_dim=1,
        slice_num=32,
        ref=8,
        n_iter=1,
        unified_pos=False,
        reshape=False,
    ):
        super(Transolver_0513_0, self).__init__()
        self.__name__ = "Transolver_0513_0"
        self.ref = ref
        self.n_layers = n_layers
        self.n_iter = n_iter
        self.unified_pos = unified_pos
        # self.reynolds_embedding = Reynolds_Embedding()
        if self.unified_pos:
            self.embedding = MLP(
                fun_dim + self.ref * self.ref * self.ref,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        else:
            self.embedding = MLP(
                fun_dim + space_dim,
                n_hidden * 2,
                n_hidden,
                n_layers=0,
                res=False,
                act=act,
            )
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Transolver_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    attn=attn,
                    attn_type=attn_type,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    last_layer=_ == n_layers - 1,
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()
        param = 1 / n_hidden * paddle.rand(shape=(n_hidden,), dtype="float32")
        out_1 = paddle.create_parameter(
            shape=param.shape,
            dtype=param.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(param),
        )
        out_1.stop_gradient = not True
        self.placeholder = out_1
        self.reshape = reshape
        if reshape:
            self.reshape_output_layer = paddle.nn.Linear(out_dim, 10)
            self.ln = paddle.nn.LayerNorm(normalized_shape=[10, 1])

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            # m.weight = ppsci.utils.initializer.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm1D)):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def reshape_output(self, x):
        x = self.reshape_output_layer(x)
        x = paddle.sum(x, axis=1, keepdim=True).transpose([0, 2, 1])
        x = self.ln(x)
        return x

    def forward(self, data):
        # 1. embedding
        # 2. model blocks
        # 3. RMSNorm
        # 4. linear
        x = self.data_dict_to_input(data)
        fx = self.embedding(x)
        fx = fx + self.placeholder[None, None, :]
        for _ in range(self.n_iter):
            for i in range(self.n_layers - 1):
                fx = self.blocks[i](fx)
        linear = self.blocks[-1]
        fx = linear(fx)
        if not self.training:
            paddle.device.cuda.empty_cache()
        if self.reshape:
            return self.reshape_output(fx)
        return fx

    def data_dict_to_input(self, inputs):
        if isinstance(inputs, list):
            # Handle list input from dataloader
            features = paddle.concat(x=inputs, axis=-1)
        else:
            # Handle dict input for backward compatibility
            x_centroid = inputs["centroids"]
            x_local_centroid = inputs["local_centroid"]
            features = paddle.concat(x=[x_centroid, x_local_centroid], axis=-1)
        return features

    @paddle.no_grad()
    def export(self, data_dict, **kwargs):
        pred = self(data_dict)
        return pred


if __name__ == "__main__":
    x, vel = paddle.ones(shape=(1, 5, 3), dtype="float32"), paddle.ones(
        shape=(1,), dtype="float32"
    )
    model = Reynolds_Embedding()
    x_new, vel_new = model(x, vel)
    print(x_new, vel_new)
    print("model loaded successfully.")
