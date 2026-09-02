from __future__ import annotations

import collections as co
import functools as ft
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generic, List,
                    Sequence, Tuple, Type, TypeVar, Union)

import paddle
import sympy

ExprType = TypeVar("ExprType", bound=sympy.Expr)
T = TypeVar("T")
if TYPE_CHECKING:
    import sympy as sympy_


def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    def fn_(*args: Any) -> T:
        return ft.reduce(fn, args)

    return fn_


def _I(*args: Any) -> paddle.Tensor:
    return paddle.tensor(1.0j)


_global_func_lookup: Dict[
    Union[Type[sympy.Basic], Callable[..., Any]], Callable[..., paddle.Tensor]
] = {
    sympy.Mul: _reduce(paddle.mul),
    sympy.Add: _reduce(paddle.add),
    sympy.div: paddle.div,
    sympy.Abs: paddle.abs,
    sympy.sign: paddle.sign,
    sympy.ceiling: paddle.ceil,
    sympy.floor: paddle.floor,
    sympy.log: paddle.log,
    sympy.exp: paddle.exp,
    sympy.sqrt: paddle.sqrt,
    sympy.cos: paddle.cos,
    sympy.acos: paddle.acos,
    sympy.sin: paddle.sin,
    sympy.asin: paddle.asin,
    sympy.tan: paddle.tan,
    sympy.atan: paddle.atan,
    sympy.atan2: paddle.atan2,
    sympy.cosh: paddle.cosh,
    sympy.acosh: paddle.acosh,
    sympy.sinh: paddle.sinh,
    sympy.asinh: paddle.asinh,
    sympy.tanh: paddle.tanh,
    sympy.atanh: paddle.atanh,
    sympy.Pow: paddle.pow,
    sympy.re: paddle.real,
    sympy.im: paddle.imag,
    sympy.arg: paddle.angle,
    sympy.erf: paddle.erf,
    sympy.loggamma: paddle.lgamma,
    sympy.Eq: paddle.eq,
    sympy.Ne: paddle.ne,
    sympy.StrictGreaterThan: paddle.gt,
    sympy.StrictLessThan: paddle.lt,
    sympy.LessThan: paddle.le,
    sympy.GreaterThan: paddle.ge,
    sympy.And: paddle.logical_and,
    sympy.Or: paddle.logical_or,
    sympy.Not: paddle.logical_not,
    sympy.Max: paddle.compat.max,
    sympy.Min: paddle.compat.min,
    sympy.MatAdd: paddle.add,
    sympy.HadamardProduct: paddle.mul,
    sympy.Trace: paddle.trace,
    sympy.Determinant: paddle.linalg.det,
    sympy.core.numbers.ImaginaryUnit: _I,
    sympy.conjugate: paddle.conj,
}
number_symbols = [cls for cls in sympy.NumberSymbol.__subclasses__()]


def number_symbol_to_paddle(symbol: sympy.NumberSymbol, *args: Any) -> paddle.Tensor:
    return paddle.tensor(float(symbol))


_global_func_lookup.update(
    {s: ft.partial(number_symbol_to_paddle, s()) for s in number_symbols}
)


class _Node(paddle.nn.Module, Generic[ExprType]):
    def __init__(
        self,
        *,
        expr: ExprType,
        _memodict: Dict[sympy.Basic, paddle.nn.Module],
        _func_lookup,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sympy_func: Type[ExprType] = expr.func
        self._paddle_func: Callable[..., paddle.Tensor]
        self._args: Union[
            paddle.nn.ModuleList,
            Tuple[Callable[[Dict[str, paddle.Tensor]], paddle.Tensor], ...],
        ]
        self._value: Any
        if issubclass(expr.func, sympy.Float):
            self._value = paddle.nn.Parameter(paddle.tensor(float(expr)))
            self._paddle_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Integer):
            self._value = paddle.tensor(int(expr))
            self._paddle_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Rational):
            self._numerator: paddle.Tensor
            self._denominator: paddle.Tensor
            assert isinstance(expr, sympy.Rational)
            self.register_buffer(
                "_numerator", paddle.tensor(expr.p, dtype=paddle.get_default_dtype())
            )
            self.register_buffer(
                "_denominator", paddle.tensor(expr.q, dtype=paddle.get_default_dtype())
            )
            self._paddle_func = lambda: self._numerator / self._denominator
            self._args = ()
        elif issubclass(expr.func, sympy.UnevaluatedExpr):
            if len(expr.args) != 1 or not issubclass(expr.args[0].func, sympy.Float):
                raise ValueError("UnevaluatedExpr should only be used to wrap floats.")
            assert isinstance(expr.args[0], sympy.Float)
            self.register_buffer("_value", paddle.tensor(float(expr.args[0])))
            self._paddle_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Symbol):
            assert isinstance(expr, sympy.Symbol)
            self._name = expr.name
            self._paddle_func = lambda value: value
            self._args = (lambda memodict: memodict[expr.name],)
        else:
            self._paddle_func = _func_lookup[expr.func]
            args: List[paddle.nn.Module] = []
            for arg in expr.args:
                try:
                    arg_ = _memodict[arg]
                except KeyError:
                    arg_ = type(self)(
                        expr=arg,
                        _memodict=_memodict,
                        _func_lookup=_func_lookup,
                        **kwargs,
                    )
                    _memodict[arg] = arg_
                args.append(arg_)
            self._args = paddle.nn.ModuleList(args)

    def sympy(self, _memodict: Dict[_Node, sympy_.Expr]) -> ExprType:
        if issubclass(self._sympy_func, sympy.Float):
            assert isinstance(self._value, paddle.nn.Parameter)
            return self._sympy_func(self._value.item())
        elif issubclass(self._sympy_func, sympy.UnevaluatedExpr):
            assert isinstance(self._value, paddle.Tensor)
            return self._sympy_func(self._value.item())
        elif issubclass(
            self._sympy_func,
            (type(sympy.S.NegativeOne), type(sympy.S.One), type(sympy.S.Zero)),
        ):
            return self._sympy_func()
        elif issubclass(self._sympy_func, sympy.Integer):
            return self._sympy_func(self._value)
        elif issubclass(self._sympy_func, sympy.Rational):
            if issubclass(self._sympy_func, type(sympy.S.Half)):
                return sympy.S.Half
            else:
                return self._sympy_func(
                    self._numerator.item(), self._denominator.item()
                )
        elif issubclass(self._sympy_func, sympy.Symbol):
            return self._sympy_func(self._name)
        elif issubclass(self._sympy_func, sympy.core.numbers.ImaginaryUnit):
            return sympy.I
        elif issubclass(self._sympy_func, sympy.core.numbers.NumberSymbol):
            return self._sympy_func()
        else:
            if issubclass(self._sympy_func, (sympy.Min, sympy.Max)):
                evaluate = False
            else:
                evaluate = True
            args = []
            for arg in self._args:
                assert isinstance(arg, _Node)
                try:
                    arg_ = _memodict[arg]
                except KeyError:
                    arg_ = arg.sympy(_memodict)
                    _memodict[arg] = arg_
                args.append(arg_)
            return self._sympy_func(*args, evaluate=evaluate)

    def forward(self, memodict) -> paddle.Tensor:
        args = []
        for arg in self._args:
            try:
                arg_ = memodict[arg]
            except KeyError:
                arg_ = arg(memodict)
                memodict[arg] = arg_
            args.append(arg_)
        return self._paddle_func(*args)


class SymPyModule(paddle.nn.Module):
    def __init__(self, *, expressions, extra_funcs=None, **kwargs):
        super().__init__(**kwargs)
        expressions = tuple(expressions)
        if extra_funcs is None:
            extra_funcs = {}
        _func_lookup = co.ChainMap(_global_func_lookup, extra_funcs)
        _memodict = {}
        self._nodes: Sequence[_Node] = paddle.nn.ModuleList(
            [
                _Node(expr=expr, _memodict=_memodict, _func_lookup=_func_lookup)
                for expr in expressions
            ]
        )
        self._expressions_string = str(expressions)

    def __repr__(self):
        return f"{type(self).__name__}(expressions={self._expressions_string})"

    def sympy(self) -> List[sympy.Expr]:
        _memodict: Dict[_Node, sympy.Expr] = {}
        return [node.sympy(_memodict) for node in self._nodes]

    def forward(self, **symbols: Any) -> paddle.Tensor:
        out = [node(symbols) for node in self._nodes]
        out = paddle.broadcast_tensors(input=out)
        return paddle.stack(out, dim=-1)
