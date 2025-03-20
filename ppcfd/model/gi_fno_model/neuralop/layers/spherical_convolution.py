import math
import warnings
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn
from scipy.optimize import brentq

from ..utils import validate_scaling_factor
from .resample import resample

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
Number = Union[int, float]


def _contract_dense(x, weight, separable=False):
    # 广播乘法 result = w.T * x
    order = len(x.shape)
    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])

    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)
    if not isinstance(weight, paddle.Tensor):
        weight = paddle.to_tensor(weight)

    return paddle.einsum(eq, x, weight)


def _contract_dense_trick(x, weight: list, separable=False):
    # the same as above function, but do the complex multiplication manually to avoid the einsum bug in paddle
    weight_real = weight[0]
    weight_imag = weight[1]
    order = len(x.shape)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)

    o1_real = paddle.einsum(eq, x.real(), weight_real) - paddle.einsum(
        eq, x.imag(), weight_imag
    )
    o1_imag = paddle.einsum(eq, x.imag(), weight_real) + paddle.einsum(
        eq, x.real(), weight_imag
    )
    x = paddle.complex(o1_real, o1_imag)
    return x


def _contract_dense_separable(x, weight, separable=True):
    if not separable:
        raise ValueError("This function is only for separable=True")
    return x * weight


def _contract_cp(x, cp_weight, separable=False):
    order = len(x.shape)
    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = (
        x_syms + "," + rank_sym + "," + ",".join(factor_syms) + "->" + "".join(out_syms)
    )

    return paddle.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = len(x.shape)
    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]  # x, y, ...
    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [
            xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])
        ]  # x, y, ...

    eq = (
        core_syms
        + ","
        + ",".join(factor_syms)
        + ","
        + x_syms
        + "->"
        + "".join(out_syms)
    )

    core = paddle.complex(tucker_weight.core_real, tucker_weight.core_imag)
    len_f = len(tucker_weight.factors_real)
    factors = [
        paddle.complex(tucker_weight.factors_real[i], tucker_weight.factors_imag[i])
        for i in range(len_f)
    ]
    out = paddle.einsum(eq, core, *factors, x)
    # Note: When exporting the model, replace the previous line of code with the line commented below
    # out = alter_einsum(eq, core, *factors, x)
    return out


def reconstruct_eq(eq1, eq2, eq_out):
    for char in eq2:
        if char not in eq1:
            eq1 += char
        elif char not in eq_out:
            eq1 = eq1.replace(char, "")
    return eq1


def alter_einsum(eq, *x):
    eq_in_lst = eq.split("->")[0].split(",")
    eq_out = eq.split("->")[-1]
    num_eqs = len(eq_in_lst)
    num_ops = len(x)
    assert (
        num_eqs == num_ops
    ), f"Invalid equation: the number of operands is {num_ops}, but found {num_eqs} segments in the label equation."

    out_last = x[0]
    eq_out_last = eq_in_lst[0]
    for i in range(1, num_ops):
        eq_out_inter = (
            reconstruct_eq(eq_out_last, eq_in_lst[i], eq_out)
            if i != num_ops - 1
            else eq_out
        )
        eq_inter = eq_out_last + "," + eq_in_lst[i] + "->" + eq_out_inter
        eq_out_last = eq_out_inter
        out_last = complex_einsum(eq_inter, out_last, x[i])
    return out_last


def complex_einsum(eq, op1, op2):
    op1_real = paddle.real(op1)
    op1_imag = paddle.imag(op1)
    op2_real = paddle.real(op2)
    op2_imag = paddle.imag(op2)

    result_real = paddle.unsqueeze(
        paddle.einsum(eq, op1_real, op2_real) - paddle.einsum(eq, op1_imag, op2_imag),
        axis=-1,
    )
    result_imag = paddle.unsqueeze(
        paddle.einsum(eq, op1_real, op2_imag) + paddle.einsum(eq, op1_imag, op2_real),
        axis=-1,
    )
    result_complex = paddle.as_complex(
        paddle.concat([result_real, result_imag], axis=-1)
    )
    return result_complex


def _contract_tt(x, tt_weight, separable=False):
    order = len(x.shape)
    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )
    return paddle.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-paddle's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            print("SEPARABLE")
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":

        if isinstance(weight, paddle.Tensor):
            return _contract_dense_trick
        elif isinstance(weight, TuckerTensor):
            return _contract_tucker
        #     if weight.name.lower() == 'complexdense':
        #         return _contract_dense
        #     elif weight.name.lower() == 'complextucker':
        #         return _contract_tucker
        #     elif weight.name.lower() == 'complextt':
        #         return _contract_tt
        #     elif weight.name.lower() == 'complexcp':
        #         return _contract_cp
        else:
            raise ValueError(f"Got unexpected factorized weight type {weight.name}")
    else:
        raise ValueError(
            f'Got implementation={implementation!r}, expected "reconstructed" or "factorized"'
        )


class FactorList(nn.Layer):
    def __init__(self, parameters=None):
        super().__init__()
        self.keys = []
        self.counter = 0
        if parameters is not None:
            self.extend(parameters)

    def _unique_key(self):
        """Creates a new unique key"""
        key = f"factor_{self.counter}"
        self.counter += 1
        return key

    def append(self, element):
        key = self._unique_key()
        if paddle.is_tensor(element):
            if isinstance(element, paddle.base.framework.EagerParamBase) or isinstance(
                element, paddle.base.framework.Parameter
            ):
                self.add_parameter(key, element)
            else:
                self.register_buffer(key, element)
        else:
            setattr(self, key, self.__class__(element))
        self.keys.append(key)

    def insert(self, index, element):
        key = self._unique_key()
        setattr(self, key, element)
        self.keys.insert(index, key)

    def pop(self, index=-1):
        item = self[index]
        self.__delitem__(index)
        return item

    def __getitem__(self, index):
        keys = self.keys[index]
        if isinstance(keys, list):
            return self.__class__([getattr(self, key) for key in keys])
        return getattr(self, keys)

    def __setitem__(self, index, value):
        setattr(self, self.keys[index], value)

    def __delitem__(self, index):
        delattr(self, self.keys[index])
        self.keys.__delitem__(index)

    def __len__(self):
        return len(self.keys)

    def extend(self, parameters):
        for param in parameters:
            self.append(param)

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __add__(self, parameters):
        instance = self.__class__(self)
        instance.extend(parameters)
        return instance

    def __radd__(self, parameters):
        instance = self.__class__(parameters)
        instance.extend(self)
        return instance

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self._parameters.items():
            size_str = "x".join(str(size) for size in p.size())
            device_str = "" if not p.is_cuda else " (GPU {})".format(p.get_device())
            parastr = "Parameter containing: [{} of size {}{}]".format(
                paddle.typename(p), size_str, device_str
            )
            child_lines.append("  (" + str(k) + "): " + parastr)
        tmpstr = "\n".join(child_lines)
        return tmpstr


class BaseSpectralConv(nn.Layer):
    def __init__(self, device=None, dtype=None):
        """Base Class for Spectral Convolutions

        Use it when you want to build your own FNO-type Neural Operators
        """
        super().__init__()

        self.dtype = dtype
        self.device = device

    def transform(self, x):
        """Transforms an input x for a skip connection, by default just an identity map

        If your function transforms the input then you should also implement this transform method
        so the skip connection can also work.

        Typical usecases are:

        * Your upsample or downsample the input in the Spectral conv: the skip connection has to be similarly scaled.
          This allows you to deal with it however you want (e.g. avoid aliasing)
        * You perform a change of basis in your Spectral Conv, again, this needs to be applied to the skip connection too.
        """
        return x


class FactorizedTensor(nn.Layer):
    def __init__(self, shape, init_scale, factorization):
        super().__init__()
        self.shape = shape
        self.init_scale = init_scale
        self.factorization = factorization
        self.real = self.create_parameter(
            shape=shape, default_initializer=nn.initializer.XavierNormal()
        )
        self.imag = self.create_parameter(
            shape=shape, default_initializer=nn.initializer.XavierNormal()
        )

    def __repr__(self):
        return f"FactorizedTensor(shape={self.shape})"

    @property
    def data(self):
        return paddle.complex(self.real, self.imag)


def validate_tucker_tensor(
    tensor_shape, rank="same", rounding="round", fixed_modes=None
):
    r"""Returns the rank of a Tucker Decomposition

    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, tuple, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple, just returns rank
    rounding = {'round', 'floor', 'ceil'}
    fixed_modes : int list or None, default is None
        if not None, a list of modes for which the rank will be the same as the original shape
        e.g. if i in fixed_modes, then rank[i] = tensor_shape[i]

    Returns
    -------
    rank : int tuple
        rank of the decomposition

    Notes
    -----
    For a fractional input rank, I want to find a Tucker rank such that:
    n_param_decomposition = rank*n_param_tensor

    In particular, for an input of size I_1, ..., I_N:
    I find a value c such that the rank will be (c I_1, ..., c I_N)

    We have sn_param_tensor = I_1 x ... x I_N

    We look for a Tucker decomposition of rank (c I_1, ..., c I_N )
    This decomposition will have the following n_params:
    For the core : \prod_k c I_k = c^N \prod I_k = c^N n_param_tensor
    For the factors : \sum_k c I_k^2

    In other words we want to solve:
    c^N n_param_tensor + \sum_k c I_k^2 = rank*n_param_tensor
    """
    if rounding == "ceil":
        rounding_fun = np.ceil
    elif rounding == "floor":
        rounding_fun = np.floor
    elif rounding == "round":
        rounding_fun = np.round
    else:
        raise ValueError(f"Rounding should be round, floor or ceil, but got {rounding}")
    # rank is 'same' or float: choose rank so as to preserve a fraction of the original #parameters
    if rank == "same":
        rank = float(1)

    if isinstance(rank, float):
        n_modes_compressed = len(tensor_shape)
        n_param_tensor = np.prod(tensor_shape)

        if fixed_modes is not None:
            tensor_shape = list(tensor_shape)

            # sorted to be careful with the order when popping and reinserting to not remove/add at wrong index.
            # list (mode, shape) that we removed as they will be kept the same, rank[i] =
            fixed_modes = [
                (mode, tensor_shape.pop(mode))
                for mode in sorted(fixed_modes, reverse=True)
            ][::-1]

            # number of parameters coming from the fixed modes (these don't have a variable size as a fun of fraction_param)
            n_fixed_params = np.sum(
                [s**2 for _, s in fixed_modes]
            )  # size of the factors
            n_modes_compressed -= len(fixed_modes)
        else:
            n_fixed_params = 0

        # Doesn't contain fixed_modes, those factors are accounted for in fixed_params
        squared_dims = np.sum([s**2 for s in tensor_shape])

        fun = (
            lambda x: n_param_tensor * x**n_modes_compressed
            + squared_dims * x
            + n_fixed_params * x
            - rank * n_param_tensor
        )
        fraction_param = brentq(fun, 0.0, max(rank, 1.0))
        rank = [max(int(rounding_fun(s * fraction_param)), 1) for s in tensor_shape]

        if fixed_modes is not None:
            for mode, size in fixed_modes:
                rank.insert(mode, size)

    elif isinstance(rank, int):
        n_modes = len(tensor_shape)
        message = f"Given only one int for 'rank' for decomposition a tensor of order {n_modes}. Using this rank for all modes."
        warnings.warn(message, RuntimeWarning)
        if fixed_modes is None:
            rank = [rank] * n_modes
        else:
            rank = [
                rank if i not in fixed_modes else s
                for (i, s) in enumerate(tensor_shape)
            ]  # *n_mode

    return rank


def _validate_tucker_tensor(tucker_tensor):
    core, factors = tucker_tensor
    if len(factors) < 2:
        raise ValueError(
            "A Tucker tensor should be composed of at least two factors and a core."
            f"However, {len(factors)} factor was given."
        )

    if len(factors) != core.ndim:
        raise ValueError(
            "Tucker decompositions should have one factor per more of the core tensor."
            f"However, core has {core.ndim} modes but {len(factors)} factors have been provided"
        )

    shape = []
    rank = []
    for i, factor in enumerate(factors):
        current_shape, current_rank = factor.shape
        if current_rank != core.shape[i]:
            raise ValueError(
                "Factor `n` of Tucker decomposition should verify:\n"
                "factors[n].shape[1] = core.shape[n]."
                f"However, factors[{i}].shape[1]={factor.shape[1]} but core.shape[{i}]={core.shape[i]}."
            )
        shape.append(current_shape)
        rank.append(current_rank)

    return tuple(shape), tuple(rank)


class TuckerTensor(nn.Layer):
    """Tucker Factorization

    Parameters
    ----------
    core
    factors
    shape
    rank
    """

    def __init__(
        self, core_real, core_imag, factors_real, factors_imag, shape=None, rank=None
    ):
        super().__init__()
        if shape is not None and rank is not None:
            self.shape, self.rank = shape, rank
        else:
            self.shape, self.rank = _validate_tucker_tensor((core_real, factors_real))

        self.order = len(self.shape)
        self.core_real = core_real
        self.core_imag = core_imag

        self.factors_real = FactorList(factors_real)
        self.factors_imag = FactorList(factors_imag)

    @classmethod
    def new(cls, shape, rank, fixed_rank_modes=None, dtype=None, **kwargs):
        # Register the parameters
        rank = validate_tucker_tensor(shape, rank, fixed_modes=fixed_rank_modes)
        core_real = paddle.create_parameter(
            rank, paddle.float32, attr=nn.initializer.Constant(value=0.0)
        )
        core_imag = paddle.create_parameter(
            rank, paddle.float32, attr=nn.initializer.Constant(value=0.0)
        )
        # Avoid the issues with ParameterList
        factors_real = [
            paddle.create_parameter(
                (s, r), paddle.float32, attr=nn.initializer.Constant(value=0.0)
            )
            for (s, r) in zip(shape, rank)
        ]
        factors_imag = [
            paddle.create_parameter(
                (s, r), paddle.float32, attr=nn.initializer.Constant(value=0.0)
            )
            for (s, r) in zip(shape, rank)
        ]
        return cls(core_real, core_imag, factors_real, factors_imag)

    def __repr__(self):
        return f"FactorizedTensor(shape={self.shape})"

    @property
    def core_data(self):
        return paddle.complex(self.core_real, self.core_imag)

    @property
    def factors_data(self):
        return paddle.complex(self.factors_real, self.factors_imag)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            mixing_factor_real, *factors_real = self.factors_real
            factors_real = [mixing_factor_real[indices, :], *factors_real]
            mixing_factors_imag, *factors_imag = self.factors_imag
            factors_imag = [mixing_factors_imag[indices, :], *factors_imag]
            return self.__class__(
                self.core_real, self.core_imag, factors_real, factors_imag
            )
        else:
            # Index multiple dimensions
            modes = []
            factors_real = []
            factors_imag = []
            factors_real_contract = []
            factors_imag_contract = []

            for i, (index, factor_real, factor_imag) in enumerate(
                zip(indices, self.factors_real, self.factors_imag)
            ):
                if index is Ellipsis:
                    raise ValueError(
                        f"Ellipsis is not yet supported, yet got indices={indices}, indices[{i}]={index}."
                    )
                if isinstance(index, int):
                    raise NotImplementedError(
                        f"Indexing with {indices} is not yet supported."
                    )
                    modes.append(i)
                    factors_real_contract.append(factor_real[index, :])
                    factors_imag_contract.append(factor_imag[index, :])
                else:
                    factors_real.append(factor_real[index, :])
                    factors_imag.append(factor_imag[index, :])

            if modes:
                raise NotImplementedError(
                    f"Indexing with {indices} is not yet supported."
                )
                # core = tenalg.multi_mode_dot(self.core, factors_contract, modes=modes)
            else:
                core_real = self.core_real
                core_imag = self.core_imag

            factors_real = factors_real + self.factors_real[i + 1 :]
            factors_imag = factors_imag + self.factors_imag[i + 1 :]

            if factors_real:
                return self.__class__(
                    core_real=core_real,
                    core_imag=core_imag,
                    factors_real=factors_real,
                    factors_imag=factors_imag,
                )
            else:
                # Fully contracted tensor
                raise NotImplementedError(
                    f"Indexing with {indices} is not yet supported."
                )
            return core_real, core_imag

    def normal_(self, mean=0, std=1):
        if mean != 0:
            raise ValueError(f"Currently only mean=0 is supported, but got mean={mean}")

        r = np.prod([math.sqrt(r) for r in self.rank])
        std_factors = (std / r) ** (1 / (self.order + 1))

        with paddle.no_grad():
            self.core_real.data.normal_(0, std_factors)
            self.core_imag.data.normal_(0, std_factors)
            for factor in self.factors_real:
                factor.data.normal_(0, std_factors)
            for factor in self.factors_imag:
                factor.data.normal_(0, std_factors)
        return self


def get_tensor_class(shape, rank, init_scale, factorization, fixed_rank_modes):
    if factorization is None:
        return FactorizedTensor(shape, init_scale, factorization)
    elif factorization == "tucker":
        return TuckerTensor.new(shape, rank, fixed_rank_modes=fixed_rank_modes)


class FactorizedSpectralConv(BaseSpectralConv):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    max_n_modes : None or int tuple, default is None
        Number of modes to use for contraction in Fourier domain during training.

        .. warning::

            We take care of the redundancy in the Fourier modes, therefore, for an input
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes
            as the real FFT is redundant along that last dimension.


        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.

        This can be updated dynamically during training.

    max_n_modes : int tuple or None, default is None
        * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
        * If None, all the n_modes are used.

    separable : bool, default is True
    init_std : float or 'auto', default is 'auto'
        std to use for the init
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    factorization : str or None, {'tucker', 'cp', 'tt'}, default is None
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor
        (vs one per layer), by default False Ignored if ``factorization is None``
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
        Ignored if ``factorization is None``
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
        Ignored if ``factorization is None``
    fft_norm : str, optional
        by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
        Ignored if ``factorization is None``
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
        Ignored if ``factorization is None``
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="backward",
        device=None,
        dtype=None,
    ):
        super().__init__(dtype=dtype, device=device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation
        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.order, n_layers)
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        else:
            init_std = init_std
        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm
        # if factorization is None:
        #     factorization = 'Dense'
        # if not factorization.lower().startswith('complex'):
        #     factorization = f'Complex{factorization}'
        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    f"To use separable Fourier Conv, in_channels must be equal to out_channels, but got in_channels={in_channels} and out_channels={out_channels}"
                )
            weight_shape = in_channels, *max_n_modes
        else:
            weight_shape = in_channels, out_channels, *max_n_modes
        self.separable = separable
        if joint_factorization:
            self.weight = paddle.create_parameter(
                shape=((2 ** (self.order - 1)) * n_layers, *weight_shape),
                dtype="float32",
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.LayerList(
                [
                    get_tensor_class(
                        weight_shape,
                        rank=self.rank,
                        init_scale=init_std,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                    )
                    for _ in range(n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)

        self._contract = get_contract_fun(
            self.weight[0], implementation=implementation, separable=separable
        )
        if bias:
            out_0 = paddle.create_parameter(
                shape=(
                    init_std
                    * paddle.randn(
                        shape=(n_layers, self.out_channels) + (1,) * self.order
                    )
                ).shape,
                dtype=(
                    init_std
                    * paddle.randn(
                        shape=(n_layers, self.out_channels) + (1,) * self.order
                    )
                )
                .numpy()
                .dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    init_std
                    * paddle.randn(
                        shape=(n_layers, self.out_channels) + (1,) * self.order
                    )
                ),
            )
            out_0.stop_gradient = not True
            self.bias = out_0
        else:
            self.bias = None

    def _get_weight(self, index):
        return self.weight[index]

    def transform(self, x, layer_index=0, output_shape=None):
        in_shape = list(tuple(x.shape)[2:])
        if self.output_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for s, r in zip(in_shape, self.output_scaling_factor[layer_index])
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape
        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
        self, x: paddle.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : paddle.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        dtype_in = x.dtype
        if self.fno_block_precision == "full":
            x = x.astype("float32")
        batchsize, channels, *mode_sizes = tuple(x.shape)
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))
        if self.fno_block_precision == "half":
            x = x.astype(dtype="float16")
        x = paddle.fft.rfftn(x=x, norm=self.fft_norm, axes=fft_dims)
        if self.order > 1:
            x = paddle.fft.fftshift(x=x, axes=fft_dims[:-1])
        if self.fno_block_precision == "mixed":
            raise NotImplementedError(
                "Mixed precision spectral conv currently unsupported"
            )

        out_fft_real = paddle.zeros(shape=[batchsize, self.out_channels, *fft_size, 1])
        out_fft_img = paddle.zeros(shape=[batchsize, self.out_channels, *fft_size, 1])
        out_fft = paddle.concat([out_fft_real, out_fft_img], axis=-1)

        # TODO: paddle.as_complex not support "float16" yet
        # if self.fno_block_precision in ["half", "mixed"]:
        #     out_fft = out_fft.astype("float16")

        out_fft = paddle.as_complex(out_fft)

        starts = [
            (max_modes - min(size, n_mode))
            for size, n_mode, max_modes in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        slices_w = [slice(None), slice(None)]
        slices_w += [
            (slice(start // 2, -start // 2) if start else slice(start, None))
            for start in starts[:-1]
        ]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = self._get_weight(indices)[slices_w]

        starts = [
            (size - min(size, n_mode))
            for size, n_mode in zip(
                list(tuple(x.shape)[2:]), list(tuple(weight.shape)[2:])
            )
        ]
        slices_x = [slice(None), slice(None)]
        slices_x += [
            (slice(start // 2, -start // 2) if start else slice(start, None))
            for start in starts[:-1]
        ]
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        # Slice currently not work
        slices_x = tuple(slices_x)
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=False)

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for s, r in zip(mode_sizes, self.output_scaling_factor[indices])
                ]
            )
        if output_shape is not None:
            mode_sizes = output_shape
        if self.order > 1:
            out_fft = paddle.fft.fftshift(x=out_fft, axes=fft_dims[:-1])
        x = paddle.fft.irfftn(
            x=out_fft, s=mode_sizes, axes=fft_dims, norm=self.fft_norm
        )
        if self.bias is not None:
            x = x + self.bias[indices, ...]
        x = x.astype(dtype_in)
        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            Warning(
                "A single convolution is parametrized, directly use the main class."
            )
        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class SubConv(paddle.nn.Layer):
    """Class representing one of the convolutions from the mother joint
    factorized convolution.

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to
    the same data, which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x, **kwargs):
        return self.main_conv.forward(x, self.indices, **kwargs)

    def transform(self, x, **kwargs):
        return self.main_conv.transform(x, self.indices, **kwargs)

    @property
    def weight(self):
        return self.main_conv.get_weight(indices=self.indices)


class FactorizedSpectralConv1d(FactorizedSpectralConv):
    def forward(self, x, indices=0):
        batchsize, channels, width = x.shape
        x = paddle.fft.rfft(x=x, norm=self.fft_norm)
        out_fft_real = paddle.zeros(
            shape=[batchsize, self.out_channels, width // 2 + 1, 1], dtype="float32"
        )
        out_fft_img = paddle.zeros(
            shape=[batchsize, self.out_channels, width // 2 + 1, 1], dtype="float32"
        )
        out_fft = paddle.concat([out_fft_real, out_fft_img], axis=-1)
        out_fft = paddle.as_complex(out_fft)

        out_fft[:, :, : self.half_n_modes[0]] = self._contract(
            x[:, :, : self.half_n_modes[0]],
            self._get_weight(indices),
            separable=self.separable,
        )
        if self.output_scaling_factor is not None:
            width = int(round(width * self.output_scaling_factor[0]))
        x = paddle.fft.irfft(x=out_fft, n=width, norm=self.fft_norm)
        if self.bias is not None:
            x = x + self.bias[indices, ...]
        return x


class FactorizedSpectralConv2d(FactorizedSpectralConv):
    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape
        x = paddle.fft.rfft2(x=x.astype(dtype="float32"), norm=self.fft_norm)
        out_fft_real = paddle.zeros(
            shape=[batchsize, self.out_channels, height, width // 2 + 1, 1],
            dtype="float32",
        )
        out_fft_img = paddle.zeros(
            shape=[batchsize, self.out_channels, height, width // 2 + 1, 1],
            dtype="float32",
        )
        out_fft = paddle.concat([out_fft_real, out_fft_img], axis=-1)
        out_fft = paddle.as_complex(out_fft)

        out_fft[:, :, : self.half_n_modes[0], : self.half_n_modes[1]] = self._contract(
            x[:, :, : self.half_n_modes[0], : self.half_n_modes[1]],
            self._get_weight(2 * indices),
            separable=self.separable,
        )
        out_fft[:, :, -self.half_n_modes[0] :, : self.half_n_modes[1]] = self._contract(
            x[:, :, -self.half_n_modes[0] :, : self.half_n_modes[1]],
            self._get_weight(2 * indices + 1),
            separable=self.separable,
        )
        if self.output_scaling_factor is not None:
            width = int(round(width * self.output_scaling_factor[indices][0]))
            height = int(round(height * self.output_scaling_factor[indices][1]))
        x = paddle.fft.irfft2(
            x=out_fft, s=(height, width), axes=(-2, -1), norm=self.fft_norm
        )
        if self.bias is not None:
            x = x + self.bias[indices, ...]
        return x


class FactorizedSpectralConv3d(FactorizedSpectralConv):
    def forward(self, x, indices=0):
        batchsize, channels, height, width, depth = x.shape
        x = paddle.fft.rfftn(
            x=x.astype(dtype="float32"), norm=self.fft_norm, axes=[-3, -2, -1]
        )
        out_fft_real = paddle.zeros(
            shape=[batchsize, self.out_channels, height, width, depth // 2 + 1, 1],
            dtype="float32",
        )
        out_fft_img = paddle.zeros(
            shape=[batchsize, self.out_channels, height, width, depth // 2 + 1, 1],
            dtype="float32",
        )
        out_fft = paddle.concat([out_fft_real, out_fft_img], axis=-1)
        out_fft = paddle.as_complex(out_fft)

        out_fft[
            :, :, : self.half_n_modes[0], : self.half_n_modes[1], : self.half_n_modes[2]
        ] = self._contract(
            x[
                :,
                :,
                : self.half_n_modes[0],
                : self.half_n_modes[1],
                : self.half_n_modes[2],
            ],
            self._get_weight(4 * indices + 0),
            separable=self.separable,
        )
        out_fft[
            :,
            :,
            : self.half_n_modes[0],
            -self.half_n_modes[1] :,
            : self.half_n_modes[2],
        ] = self._contract(
            x[
                :,
                :,
                : self.half_n_modes[0],
                -self.half_n_modes[1] :,
                : self.half_n_modes[2],
            ],
            self._get_weight(4 * indices + 1),
            separable=self.separable,
        )
        out_fft[
            :,
            :,
            -self.half_n_modes[0] :,
            : self.half_n_modes[1],
            : self.half_n_modes[2],
        ] = self._contract(
            x[
                :,
                :,
                -self.half_n_modes[0] :,
                : self.half_n_modes[1],
                : self.half_n_modes[2],
            ],
            self._get_weight(4 * indices + 2),
            separable=self.separable,
        )
        out_fft[
            :,
            :,
            -self.half_n_modes[0] :,
            -self.half_n_modes[1] :,
            : self.half_n_modes[2],
        ] = self._contract(
            x[
                :,
                :,
                -self.half_n_modes[0] :,
                -self.half_n_modes[1] :,
                : self.half_n_modes[2],
            ],
            self._get_weight(4 * indices + 3),
            separable=self.separable,
        )
        if self.output_scaling_factor is not None:
            width = int(round(width * self.output_scaling_factor[0]))
            height = int(round(height * self.output_scaling_factor[1]))
            depth = int(round(depth * self.output_scaling_factor[2]))
        x = paddle.fft.irfftn(x=out_fft, s=(height, width, depth), norm=self.fft_norm)
        if self.bias is not None:
            x = x + self.bias[indices, ...]
        return x


if __name__ == "__main__":
    # let x be a complex tensor of size (32, 32, 8, 8)
    x = paddle.randn([32, 32, 8, 8]).astype("complex64")
    # let weight be the same
    weight = paddle.randn([32, 32, 8, 8]).astype("complex64")
    weight = paddle.randn([32, 32, 8, 8]).astype("complex64")
    weight = paddle.create_parameter(
        weight.shape, weight.dtype, attr=paddle.nn.initializer.Assign(weight)
    )

    separable = False
    result = _contract_dense(x, weight, separable=separable)
    print(result)
