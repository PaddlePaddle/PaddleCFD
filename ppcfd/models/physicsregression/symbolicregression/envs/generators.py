import copy
import json
import math
import random
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.special
import sympy as sp
from scipy.stats import special_ortho_group
from timeout_decorator import timeout

from ..._paths import FEYNMAN_EQUATIONS_PATH
from ..._paths import UNITS_CSV_PATH
from . import encoders
from .node import Node
from .operators import *


class Generator(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def generate_datapoints(self, rng):
        pass


class RandomFunctions(Generator):
    def __init__(self, params, special_words):
        super().__init__(params)
        self.params = params
        self.prob_const = params.prob_const
        self.prob_rand = params.prob_rand
        self.max_int = params.max_int
        self.min_binary_ops_per_dim = params.min_binary_ops_per_dim
        self.max_binary_ops_per_dim = params.max_binary_ops_per_dim
        self.min_unary_ops = params.min_unary_ops
        self.max_unary_ops = params.max_unary_ops
        self.min_output_dimension = params.min_output_dimension
        self.min_input_dimension = params.min_input_dimension
        self.max_input_dimension = params.max_input_dimension
        self.max_output_dimension = params.max_output_dimension
        self.max_number = 10 ** (params.max_exponent + params.float_precision)
        self.operators = copy.deepcopy(operators_real)
        self.operators_dowsample_ratio = defaultdict(float)
        if params.operators_to_downsample != "":
            for operator in self.params.operators_to_downsample.split(","):
                operator, ratio = operator.split("_")
                ratio = float(ratio)
                self.operators_dowsample_ratio[operator] = ratio
        if params.required_operators != "":
            self.required_operators = self.params.required_operators.split(",")
        else:
            self.required_operators = []
        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []
        self.unaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 1
        ] + self.extra_unary_operators
        self.binaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 2
        ] + self.extra_binary_operators
        unaries_probabilities = []
        for op in self.unaries:
            if op not in self.operators_dowsample_ratio:
                unaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                unaries_probabilities.append(ratio)
        self.unaries_probabilities = np.array(unaries_probabilities)
        self.unaries_probabilities /= self.unaries_probabilities.sum()
        binaries_probabilities = []
        for op in self.binaries:
            if op not in self.operators_dowsample_ratio:
                binaries_probabilities.append(1.0)
            else:
                ratio = self.operators_dowsample_ratio[op]
                binaries_probabilities.append(ratio)
        self.binaries_probabilities = np.array(binaries_probabilities)
        self.binaries_probabilities /= self.binaries_probabilities.sum()
        self.unary = False
        self.constants = [
            str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants
        self.variables = (
            ["rand"] + [f"x_{i}" for i in range(self.max_input_dimension)] + ["y"]
        )
        self.symbols = (
            list(self.operators)
            + self.constants
            + self.variables
            + ["|", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
        )
        self.constants.remove("CONSTANT")
        if self.params.extra_constants is not None:
            self.extra_constants = self.params.extra_constants.split(",")
        else:
            self.extra_constants = []
        self.general_encoder = encoders.GeneralEncoder(
            params, self.symbols, all_operators
        )
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words
        if self.params.dim_length == 2:
            units = [
                ("M" + str(i) + "S" + str(j))
                for i in range(-20, 21)
                for j in range(-20, 21)
            ] + [
                ("K" + str(i) + "T" + str(j) + "V" + str(k))
                for i in range(-9, 10)
                for j in range(-5, 6)
                for k in range(-9, 10)
            ]
        elif self.params.dim_length == 5:
            units = (
                [("M" + str(i)) for i in range(-30, 31)]
                + [("S" + str(i)) for i in range(-30, 31)]
                + [("K" + str(i)) for i in range(-30, 31)]
                + [("T" + str(i)) for i in range(-20, 21)]
                + [("V" + str(i)) for i in range(-20, 21)]
            )
        else:
            raise ValueError(f"Error dim lenght: {self.params.dim_length}")
        comp = [f"COMPLEXITY:{i}" for i in ["simple", "middle", "hard"]]
        self.equation_words += units
        self.equation_words += comp
        if params.expr_train_data_path:
            with open(params.expr_train_data_path, "r") as fi:
                self.exprs_train = json.load(fi)
        else:
            self.exprs_train = None
        if params.expr_valid_data_path:
            with open(params.expr_valid_data_path, "r") as fi:
                self.exprs_valid = json.load(fi)
        else:
            self.exprs_valid = None
        if params.expr_test_data_path:
            with open(params.expr_test_data_path, "r") as fi:
                self.exprs_test = json.load(fi)
        else:
            self.exprs_test = None
        self.testing_exprs_idx = params.eval_start_from
        if self.params.use_exprs > -1:
            self.exprs_train = self.exprs_train[: self.params.use_exprs]
        if params.sub_expr_train_path:
            with open(params.sub_expr_train_path, "r") as fi:
                self.sub_exprs_train = json.load(fi)
        else:
            self.sub_exprs_train = None
        if params.sub_expr_valid_path:
            with open(params.sub_expr_valid_path, "r") as fi:
                self.sub_exprs_valid = json.load(fi)
        else:
            self.sub_exprs_valid = None
        if params.pre_differentiate_path:
            with open(params.pre_differentiate_path, "r") as fi:
                self.differentiation_dic = json.load(fi)
        else:
            self.differentiation_dic = {}
        feynman_path = getattr(self.params, "feynman_equations_path", None)
        df = pd.read_excel(feynman_path or FEYNMAN_EQUATIONS_PATH)
        self.feynman_exprs = list(df["Formula"])
        self.feynman_exprs_idx = params.eval_start_from
        self.to_sample_expr = []
        units_path = getattr(self.params, "units_csv_path", None)
        df = pd.read_csv(units_path or UNITS_CSV_PATH)
        df = df[:-1]
        self.physical_units_dic = {
            variable: np.array([m, s, kg, T, V])
            for variable, m, s, kg, T, V in zip(
                df["Variable"], df["m"], df["s"], df["kg"], df["T"], df["V"]
            )
        }
        self.physical_units_dic["pi"] = np.zeros(5)
        self.physical_units_dic["E"] = np.zeros(5)
        for i in range(10):
            self.physical_units_dic[f"x_{i}"] = np.zeros(5)
        for i in range(3, 6):
            self.physical_units_dic[f"theta{i}"] = np.zeros(5)
        self.physical_units_dic["dummy"] = np.zeros(5)
        self.variable_list = list(df["Variable"][:-1])
        self.variable_list = [sp.symbols(v) for v in self.variable_list]
        self.binary = ["add", "sub", "mul", "pow"]
        self.unary = [
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "arcsin",
            "arccos",
            "neg",
            "arctan",
            "inv",
            "pow2",
            "pow3",
            "pow4",
            "pow5",
            "tanh",
            "sinh",
            "cosh",
            "abs",
        ]
        self.locals = ["gamma", "beta", "theta", "rf", "Ei"]
        self.locals += [chr(65 + i) for i in range(26)]
        self.locals = set(self.locals)
        self.locals = {v: sp.symbols(v) for v in self.locals}
        self.binary_complex_dic = binary_complex_dic
        self.unary_complex_dic = unary_complex_dic

    def add_random_const(self, rng, expr, p_add=0.15, p_mul=0.12):
        expr = sp.parse_expr(expr, local_dict=self.locals)

        def _add_random_const_strategy1(expr: sp.Expr, type: str, const: float):
            """
            Param:
            ------------
            `expr`: sp.Expr
            `type`: "add" or "mul"
            `const`: float

            Return:
            ------------
            return a sp.Expr of (const +/* expr)
            """
            args = sp.simplify(const), expr
            if type == "add":
                new_expr = sp.Add(*args)
                return new_expr
            elif type == "mul":
                new_expr = sp.Mul(*args)
                return new_expr
            else:
                raise TypeError

        def _add_random_const_strategy2(
            expr: sp.Expr, args: tuple, type: str, const: float
        ):
            """
            Param:
            ------------
            `expr`: sp.Expr
            `args`: typle(sp.Expr)
            `type`: "add" or "mul"
            `const`: float

            Return:
            ------------
            return a sp.Expr of (const +/* args)
            """
            args = sp.simplify(const), *expr.args
            if type == "add":
                assert isinstance(expr, sp.core.add.Add), f"error with expr type:{expr}"
                new_expr = sp.Add(*args)
                return new_expr
            elif type == "mul":
                assert isinstance(expr, sp.core.mul.Mul), f"error with expr type:{expr}"
                new_expr = sp.Mul(*args)
                return new_expr
            else:
                raise TypeError

        def _construct_expr(args, expr):
            """
            Params:
            --------
            `args`: arguments of exprssion to be construct
            `expr`: the operator of top node

            """
            if isinstance(expr, sp.core.add.Add):
                return sp.Add(*args)
            elif isinstance(expr, sp.core.mul.Mul):
                return sp.Mul(*args)
            elif isinstance(expr, sp.core.power.Pow):
                return sp.Pow(*args)
            elif isinstance(expr, sp.exp):
                return sp.exp(*args)
            elif isinstance(expr, sp.log):
                return sp.log(*args)
            elif isinstance(expr, sp.sin):
                return sp.sin(*args)
            elif isinstance(expr, sp.cos):
                return sp.cos(*args)
            elif isinstance(expr, sp.tan):
                return sp.tan(*args)
            elif isinstance(expr, sp.asin):
                return sp.asin(*args)
            elif isinstance(expr, sp.acos):
                return sp.acos(*args)
            elif isinstance(expr, sp.atan):
                return sp.atan(*args)
            else:
                return expr

        def _recursion(rng, expr: sp.Expr):
            args = []
            if isinstance(expr, sp.core.power.Pow):
                args = [_recursion(rng, expr.args[0]), expr.args[1]]
            elif isinstance(expr, sp.core.mul.Mul):
                for arg in expr.args:
                    if not isinstance(arg, sp.core.power.Pow):
                        args.append(_recursion(rng, arg))
                    else:
                        args.append(sp.Pow(_recursion(rng, arg.args[0]), arg.args[1]))
            else:
                for arg in expr.args:
                    args.append(_recursion(rng, arg))
            p_a = rng.random()
            p_m = rng.random()
            if p_m < p_mul and not isinstance(expr, sp.core.numbers.Number):
                mul_const = rng.normal(0, 1) + (rng.randint(0, 2) * 2 - 1)
                if isinstance(expr, sp.core.mul.Mul):
                    expr = _add_random_const_strategy2(expr, args, "mul", mul_const)
                else:
                    expr = _construct_expr(args, expr)
                    expr = _add_random_const_strategy1(expr, "mul", mul_const)
            elif (
                p_a < p_add
                and not isinstance(expr, sp.core.numbers.Pi)
                and not isinstance(expr, sp.core.numbers.Number)
            ):
                add_const = rng.normal(0, 1)
                if isinstance(expr, sp.core.add.Add):
                    expr = _add_random_const_strategy2(expr, args, "add", add_const)
                else:
                    expr = _construct_expr(args, expr)
                    expr = _add_random_const_strategy1(expr, "add", add_const)
            else:
                expr = _construct_expr(args, expr)
            return expr

        return str(_recursion(rng, expr))

    def infix_to_prefix(self, infix, sp_parse=True, replace_op_dic={}):
        def _infix_to_prefix(lis):
            for i in range(len(lis) - 1, -1, -1):
                if lis[i] in self.unary and lis[i] not in ["neg", "inv"]:
                    lis[i] = lis[i] + "," + lis[i + 1]
                    lis.pop(i + 1)
            if "add" in lis or "sub" in lis:
                idx = [i for i in range(len(lis)) if lis[i] == "add" or lis[i] == "sub"]
                idx = idx[-1]
                return (
                    lis[idx]
                    + ","
                    + _infix_to_prefix(lis[:idx])
                    + ","
                    + _infix_to_prefix(lis[idx + 1 :])
                )
            elif "neg" == lis[0]:
                return lis[0] + "," + _infix_to_prefix(lis[1:])
            elif "mul" in lis:
                idx = [i for i in range(len(lis)) if lis[i] in ["mul"]]
                idx = idx[0]
                return (
                    lis[idx]
                    + ","
                    + _infix_to_prefix(lis[:idx])
                    + ","
                    + _infix_to_prefix(lis[idx + 1 :])
                )
            elif "inv" == lis[0]:
                return lis[0] + "," + _infix_to_prefix(lis[1:])
            elif "pow" in lis:
                idx = [i for i in range(len(lis)) if lis[i] in ["pow"]]
                idx = idx[-1]
                return (
                    lis[idx]
                    + ","
                    + _infix_to_prefix(lis[:idx])
                    + ","
                    + _infix_to_prefix(lis[idx + 1 :])
                )
            else:
                return ",".join(lis)

        if sp_parse:
            infix = self.parse_sympy_expr(infix)
            infix = str(infix)
        infix = infix.replace(" ", "")
        infix = infix.replace("**", "^")
        infix = infix.replace("ln", "log")
        infix = infix.replace("Abs", "abs")
        infix_stack = []
        temp = ""
        for i in range(len(infix)):
            if (
                infix[i] in ["(", ")", "*", "/", "^"]
                or infix[i] in ["-", "+"]
                and infix[i - 1] != "e"
            ):
                if temp:
                    infix_stack.append(temp)
                    temp = ""
                infix_stack.append(infix[i])
            else:
                temp += infix[i]
        if temp:
            infix_stack.append(temp)
        infix_stack = ["("] + infix_stack + [")"]
        iinfix_stack = []
        for i in range(len(infix_stack)):
            if i < len(infix_stack) - 1:
                is_float_next, f = self.is_float(infix_stack[i + 1])
            else:
                is_float = False
            if (
                infix_stack[i] == "-"
                and is_float_next
                and i > 0
                and infix_stack[i - 1] in ["+", "-", "*", "(", "/", "^"]
            ):
                infix_stack[i + 1] = str(-f)
            elif infix_stack[i] == "-" and infix_stack[i - 1] == "(":
                iinfix_stack.append("neg")
            elif infix_stack[i] == "+":
                iinfix_stack.append("add")
            elif infix_stack[i] == "-":
                iinfix_stack.append("sub")
            elif infix_stack[i] == "*":
                iinfix_stack.append("mul")
            elif infix_stack[i] == "/":
                iinfix_stack.append("mul")
                iinfix_stack.append("inv")
            elif infix_stack[i] == "^":
                iinfix_stack.append("pow")
            else:
                iinfix_stack.append(infix_stack[i])
        infix_stack = iinfix_stack
        for i in range(len(infix_stack)):
            infix_stack[i] = replace_op_dic.get(infix_stack[i], infix_stack[i])
        stack = []
        for op in infix_stack:
            if op == ")":
                idx = [i for i in range(len(stack)) if stack[i] == "("]
                idx = idx[-1]
                prefix = _infix_to_prefix(stack[idx + 1 :])
                stack = stack[:idx] + [prefix]
            else:
                stack.append(op)
        prefix = stack[0]
        prefix = prefix.split(",")
        return prefix

    def prefix_to_node(self, rng, prefix, variables=None, random_variables_seq=False):
        if variables is None:
            variables = []
            for i in range(len(prefix)):
                op = prefix[i]
                try:
                    float(op)
                except:
                    if op not in self.binary + self.unary + math_constants:
                        if op not in variables:
                            variables.append(op)
        if random_variables_seq:
            rng.shuffle(variables)
        variables_dic = {v: f"x_{i}" for i, v in enumerate(variables)}
        prefix = [variables_dic.get(p, p) for p in prefix]

        def _prefix_to_node(prefix):
            if prefix == "":
                return
            prefix = prefix.split(",")
            op = prefix[0]
            if op[:2] == "x_":
                node = Node(op, self.params)
                return node, ",".join(prefix[1:])
            try:
                _ = float(op)
                node = Node(op, self.params)
                return node, ",".join(prefix[1:])
            except:
                pass
            if op in self.unary:
                node = Node(op, self.params)
                child_node, prefix = _prefix_to_node(",".join(prefix[1:]))
                node.push_child(child_node)
                return node, prefix
            elif op in self.binary:
                node = Node(op, self.params)
                child_node1, prefix1 = _prefix_to_node(",".join(prefix[1:]))
                node.push_child(child_node1)
                child_node2, prefix2 = _prefix_to_node(prefix1[:])
                node.push_child(child_node2)
                return node, prefix2
            elif op in math_constants:
                if op == "pi":
                    node = Node("pi", self.params)
                    return node, ",".join(prefix[1:])
                elif op == "E":
                    node = Node("E", self.params)
                    return node, ",".join(prefix[1:])
                else:
                    print(f"error op:{op}")
                    raise ValueError(f"error op:{op}")
            else:
                print("error, unseen op:" + op)
                raise ValueError("error, unseen op:" + op)
            return node, prefix[1:]

        node, _ = _prefix_to_node(",".join(prefix))
        return node, variables

    def post_process(self, node, allow_pow=False):
        stack = [node]
        while stack:
            temp = stack.pop(0)
            if temp.value == "pow":
                try:
                    f = int(temp.children[1].value)
                    assert 2 <= f <= 5
                    temp.value = "pow" + str(f)
                    temp.children = [temp.children[0]]
                except:
                    assert (
                        allow_pow
                    ), f"not allowed in pow {temp.children[1]}, not a integer"
            elif temp.value == "neg":
                try:
                    f = float(temp.children[0].value)
                    is_const = True
                except:
                    is_const = False
                if is_const:
                    temp.value = str(-f)
                    temp.children = []
            stack += temp.children
        return node

    def infix_to_node(
        self,
        infix,
        variables=None,
        allow_pow=False,
        label_units=True,
        rng=np.random,
        random_variables_seq=False,
        sp_parse=True,
        replace_op_dic={},
    ):
        prefix = self.infix_to_prefix(infix, sp_parse, replace_op_dic)
        node, real_variables = self.prefix_to_node(
            rng, prefix, variables, random_variables_seq
        )
        node = self.post_process(node, allow_pow)
        if label_units:
            self.label_units(node, variables=real_variables)
        return node, real_variables

    def _dfs_units(self, node, tgt_prefix_units=[], variable_units=[]):
        op = node.value
        children = node.children
        use_tgt_units = True if tgt_prefix_units else False
        if len(children) == 0:
            if not use_tgt_units and not variable_units:
                return
            tgt_unit = tgt_prefix_units.pop(0) if use_tgt_units else None
            if isinstance(op, str) and op[:2] == "x_":
                node.unit = tgt_unit if use_tgt_units else variable_units[int(op[2:])]
                return
            try:
                _ = float(op)
                node.unit = tgt_unit if use_tgt_units else np.zeros(5)
            except:
                node.unit = tgt_unit if use_tgt_units else np.zeros(5)
            return
        elif op in self.unary:
            tgt_unit = tgt_prefix_units.pop(0) if use_tgt_units else None
            self._dfs_units(children[0], tgt_prefix_units, variable_units)
            if op == "inv":
                node.unit = -children[0].unit
            elif op == "sqrt":
                node.unit = children[0].unit / 2
            elif op == "neg":
                node.unit = children[0].unit
            elif op.startswith("pow"):
                node.unit = children[0].unit * int(op[3])
            elif op == "abs":
                node.unit = children[0].unit
            else:
                assert all(children[0].unit == 0)
                node.unit = children[0].unit
            if use_tgt_units:
                assert all(tgt_unit == node.unit)
        elif op in self.binary:
            tgt_unit = tgt_prefix_units.pop(0) if use_tgt_units else None
            self._dfs_units(children[0], tgt_prefix_units, variable_units)
            self._dfs_units(children[1], tgt_prefix_units, variable_units)
            if op in ["add", "sub"]:
                try:
                    _ = float(children[0].value)
                    left_const = True
                except:
                    left_const = False
                try:
                    _ = float(children[1].value)
                    right_const = True
                except:
                    right_const = False
                if left_const:
                    children[0].unit = children[1].unit
                    node.unit = children[1].unit
                elif right_const:
                    children[1].unit = children[0].unit
                    node.unit = children[0].unit
                else:
                    assert all(children[0].unit == children[1].unit)
                    node.unit = children[0].unit
            elif op == "mul":
                node.unit = children[0].unit + children[1].unit
            elif op == "div":
                node.unit = children[0].unit - children[1].unit
            elif op == "pow":
                try:
                    int_value = int(float(children[1].value))
                    is_int = True
                except:
                    is_int = False
                    raise NotImplementedError(
                        f"unsupport operator of pow for labeling dimension!"
                    )
                node.unit = children[0].unit * int_value
            else:
                raise NotImplementedError(f"unseen binary operator: {op}")
            if use_tgt_units:
                assert all(tgt_unit == node.unit)
        elif op in math_constants:
            tgt_unit = tgt_prefix_units.pop(0) if use_tgt_units else None
            if op == "pi":
                node.unit = tgt_unit if use_tgt_units else np.zeros(5)
            elif op == "E":
                node.unit = tgt_unit if use_tgt_units else np.zeros(5)
            else:
                raise NotImplementedError(f"unseen math constant operator: {op}")
        else:
            raise NotImplementedError(f"unseen operator: {op}")

    def label_units(self, tree, variables=None, units=None):
        assert variables is not None or units is not None
        if variables is not None:
            units = [self.physical_units_dic[x] for x in variables]
        elif units is not None:
            units = units
        self._dfs_units(tree, variable_units=units)

    def label_units_with_consts(self, tree, units_prefix):
        assert len(units_prefix) == len(tree)
        tgt_prefix_units = []
        for i in range(len(units_prefix)):
            tgt_prefix_units.append(np.array(units_prefix[i]))
        self._dfs_units(tree, tgt_prefix_units=tgt_prefix_units)

    def generate_multi_dimensional_tree(self, rng, expr=None, datatype="train"):
        """rng_state = rng.get_state()
        seed = rng_state[1][0]"""
        units = None
        consts = None
        c_units = None
        idx = 0
        if len(self.to_sample_expr) != 0:
            expr, idx = self.to_sample_expr[0]
        elif expr is not None and expr == "feynman":
            idx = self.feynman_exprs_idx
            expr = self.feynman_exprs[idx]
            expr = expr.replace("COS", "cos")
            self.feynman_exprs_idx = (self.feynman_exprs_idx + 1) % len(
                self.feynman_exprs
            )
        elif expr is not None:
            pass
        elif datatype == "train":
            if self.params.sub_expr_train_path and rng.random() < 0.3:
                idx = rng.randint(0, len(self.sub_exprs_train))
                expr, units, consts, c_units = self.sub_exprs_train[idx].values()
            else:
                idx = rng.randint(0, len(self.exprs_train))
                expr = self.exprs_train[idx]
        elif datatype == "valid":
            idx = rng.randint(0, len(self.exprs_valid))
            expr = self.exprs_valid[idx]
        elif datatype == "valid-sub":
            idx = rng.randint(0, len(self.sub_exprs_valid))
            expr, units, consts, c_units = self.sub_exprs_valid[idx].values()
        elif datatype == "test":
            idx = self.testing_exprs_idx
            expr = self.exprs_test[idx]
            self.testing_exprs_idx = (self.testing_exprs_idx + 1) % len(self.exprs_test)
        else:
            raise ValueError()
        p_add = self.params.p_add
        p_mul = self.params.p_mul
        if self.params.add_consts == 1 and (
            datatype == "train" and units is None or datatype == "test"
        ):
            infix = self.add_random_const(rng, expr, p_add, p_mul)
        else:
            infix = expr
        node, real_variables = self.infix_to_node(
            infix,
            rng=rng,
            label_units=True if units is None else False,
            random_variables_seq=self.params.random_variables_sequence,
            sp_parse=True if units is None else False,
            allow_pow=False if datatype == "train" else True,
        )
        if units is not None:
            self.label_units_with_consts(node, units)
        if self.params.add_consts == 1 and datatype == "train" and units is not None:
            if node.value != "add" and rng.random() < 0.2:
                root = Node("add", self.params)
                const = Node(str(rng.normal(0, 1)), self.params)
                root.children = [node, const]
                root.unit = node.unit
                const.unit = node.unit
                node = root
        if self.params.sample_expr_num > 1 and len(self.to_sample_expr) == 0:
            self.to_sample_expr.extend(
                [[expr, idx]] * (self.params.sample_expr_num - 1)
            )
        elif len(self.to_sample_expr) != 0:
            self.to_sample_expr.pop(0)
        if rng.random() < 0.2 and datatype == "train":
            dfs_stack = [node]
            freq0 = rng.uniform(0.1, 6) * (rng.randint(0, 2) * 2 - 1)
            use_same_freq = 1 if rng.random() < 0.3 else 0
            while dfs_stack:
                current_node = dfs_stack.pop(0)
                if current_node.value in ["sin", "cos"] and len(current_node) == 2:
                    if current_node.children[0].value.startswith("x_"):
                        if use_same_freq:
                            freq = freq0
                        else:
                            freq = rng.uniform(0.1, 6) * (rng.randint(0, 2) * 2 - 1)
                        node2 = Node(str(freq), self.params)
                        node1 = Node(
                            "mul",
                            self.params,
                            children=[node2, current_node.children[0]],
                        )
                        current_node.children[0] = node1
                        node1.unit = np.zeros(5)
                        node2.unit = np.zeros(5)
                else:
                    dfs_stack.extend(current_node.children)
        num_variables = len(real_variables)
        unary_ops_to_use = [
            [x for x in tree_i.prefix().split(",") if x in self.unaries]
            for tree_i in [node]
        ]
        binary_ops_to_use = [
            [x for x in tree_i.prefix().split(",") if x in self.binaries]
            for tree_i in [node]
        ]
        return (
            expr,
            node,
            num_variables,
            1,
            unary_ops_to_use,
            binary_ops_to_use,
            real_variables,
            idx,
            consts,
            c_units,
        )

    def differentiate(self, tree, original_expr, input_dimension, structure_type):
        assert structure_type in [
            "add",
            "mul",
        ], f"wrong structure type: {structure_type}!"
        if original_expr in self.differentiation_dic:
            i = (structure_type == "mul") * 1
            seperate_idx = self.differentiation_dic[original_expr][i]
            return seperate_idx
        symbol_x = [sp.symbols(f"x_{i}") for i in range(input_dimension)]
        expr = tree.infix()
        for op, op_new in zip(
            ["add", "sub", "mul", "div", "pow", "inv", "arcsin", "arccos", "arctan"],
            ["+", "-", "*", "/", "**", "1/", "asin", "acos", "atan"],
        ):
            expr = expr.replace(op, op_new)
        if structure_type == "add":
            pass
        elif structure_type == "mul":
            expr = "log(" + expr + ")"
        expr = sp.parse_expr(expr)
        seperate_idx = []
        for i in range(input_dimension):
            first_derivative = sp.diff(expr, symbol_x[i]).cancel()
            for j in range(i + 1, input_dimension):
                second_derivative = sp.diff(first_derivative, symbol_x[j]).cancel()
                if (
                    isinstance(second_derivative, sp.core.numbers.Number)
                    and abs(second_derivative) < 1e-05
                ):
                    seperate_idx.append((i, j))
        return seperate_idx

    def sample_differentiation(self, rng, seperate_idx, sample_prob):
        return [idx for idx in seperate_idx if rng.random() < sample_prob]

    def find_expr_structure(self, rng, seperate_idx, input_dimension):
        groups = [np.arange(input_dimension)]
        for idx in seperate_idx:
            for i in range(len(groups)):
                idx0 = groups[i] == idx[0]
                idx1 = groups[i] == idx[1]
                if any(idx0) and any(idx1):
                    groups0 = groups[i][~idx0]
                    groups1 = groups[i][~idx1]
                    groups[i] = groups0
                    groups.append(groups1)
        idx_to_delete = []
        for i in range(len(groups)):
            for j in range(len(groups)):
                if i == j:
                    continue
                if all([(ele in groups[j]) for ele in groups[i]]):
                    if len(groups[i]) == len(groups[j]):
                        idx_to_delete.append(max(i, j))
                    else:
                        idx_to_delete.append(i)
        return [groups[i] for i in range(len(groups)) if i not in idx_to_delete]

    def compute_complexity1(self, tree):
        def _dfs(node):
            if len(node.children) == 0 and node.value.startswith("x_"):
                node.complexity = 1
                return
            elif len(node.children) == 0:
                node.complexity = 1
                return
            for child in node.children:
                _dfs(child)
            if node.value in self.binary_complex_dic:
                node.complexity = (
                    node.children[0].complexity + node.children[1].complexity
                )
            elif node.value in self.unary_complex_dic:
                node.complexity = (
                    self.unary_complex_dic[node.value] * node.children[0].complexity
                )
            else:
                raise NotImplementedError

        _dfs(tree)
        return tree.complexity

    def compute_complexity2(self, tree):
        def _dfs(node):
            node.complexity = len(node)
            for child in node.children:
                _dfs(child)

        _dfs(tree)
        return tree.complexity

    def is_float(self, value, allow_int=True):
        try:
            f = float(value)
            is_f = True if allow_int else int(f) != f
        except:
            f = None
            is_f = False
        return is_f, f

    def reduce_const(self, tree):
        for node in tree.children:
            self.reduce_const(node)
        if tree.value in self.unary:
            is_float, f = self.is_float(tree.children[0].value)
            if is_float:
                tree.value = tree.val(np.zeros(1)).item()
                tree.children = []
        elif tree.value in self.binary:
            is_float_left, f_left = self.is_float(tree.children[0].value)
            is_float_right, f_right = self.is_float(tree.children[1].value)
            if is_float_left and is_float_right:
                tree.value = tree.val(np.zeros(1)).item()
                tree.children = []

    def swap_node(self, tree):
        swap_flag = False
        for node in tree.children:
            swap_flag = swap_flag or self.swap_node(node)
        if tree.value in ["add", "sub"]:
            is_float_left, _ = self.is_float(tree.children[0].value)
            is_float_right, _ = self.is_float(tree.children[1].value)
            if not is_float_left and tree.children[1].value in ["add", "sub"]:
                is_float_left, _ = self.is_float(tree.children[1].children[0].value)
                is_float_right, _ = self.is_float(tree.children[1].children[1].value)
                if is_float_left:
                    children = tree.children[1].children[0]
                    tree.children[1].children[0] = tree.children[0]
                    tree.children[0] = children
                    if tree.value == "sub" and tree.children[1].value == "add":
                        tree.children[0].value = str(-float(tree.children[0].value))
                        tree.value = "add"
                        tree.children[1].value = "sub"
                    elif tree.value == "sub" and tree.children[1].value == "sub":
                        tree.children[0].value = str(-float(tree.children[0].value))
                        tree.value = "add"
                        tree.children[1].value = "add"
                    return True
                elif is_float_right:
                    children = tree.children[1].children[1]
                    tree.children[1].children[1] = tree.children[0]
                    tree.children[0] = children
                    if tree.value == "sub" and tree.children[1].value == "add":
                        tree.children[0].value = str(-float(tree.children[0].value))
                        tree.children[1].value = "sub"
                    elif tree.value == "add" and tree.children[1].value == "sub":
                        tree.children[0].value = str(-float(tree.children[0].value))
                        tree.children[1].value = "add"
                    return True
            elif is_float_right:
                children = tree.children[0]
                tree.children[0] = tree.children[1]
                tree.children[1] = children
                if tree.value == "sub":
                    tree.value = "add"
                    tree.children[0].value = str(-float(tree.children[0].value))
                return True
        elif tree.value in ["mul"]:
            is_float_left, _ = self.is_float(tree.children[0].value)
            is_float_right, _ = self.is_float(tree.children[1].value)
            if not is_float_left and tree.children[1].value in ["mul"]:
                is_float_left, _ = self.is_float(tree.children[1].children[0].value)
                is_float_right, _ = self.is_float(tree.children[1].children[1].value)
                if is_float_left:
                    children = tree.children[1].children[0]
                    tree.children[1].children[0] = tree.children[0]
                    tree.children[0] = children
                    tree.children[1].unit = (
                        tree.children[1].children[0].unit
                        + tree.children[1].children[1].unit
                    )
                    tree.unit = tree.children[0].unit + tree.children[1].unit
                    return True
                elif is_float_right:
                    children = tree.children[1].children[1]
                    tree.children[1].children[1] = tree.children[0]
                    tree.children[0] = children
                    tree.children[1].unit = (
                        tree.children[1].children[0].unit
                        + tree.children[1].children[1].unit
                    )
                    tree.unit = tree.children[0].unit + tree.children[1].unit
                    return True
            elif is_float_right:
                children = tree.children[0]
                tree.children[0] = tree.children[1]
                tree.children[1] = children
                return True
        return swap_flag

    def partial_reduce_const(self, tree):
        for node in tree.children:
            self.partial_reduce_const(node)
        if tree.value == "add" and tree.children[1].value == "add":
            is_f1, f1 = self.is_float(tree.children[0].value)
            is_f2, f2 = self.is_float(tree.children[1].children[0].value)
            if is_f1 and is_f2:
                tree.children[1] = tree.children[1].children[1]
                tree.children[0].value = f1 + f2
        elif tree.value == "add" and tree.children[1].value == "sub":
            is_f1, f1 = self.is_float(tree.children[0].value)
            is_f2, f2 = self.is_float(tree.children[1].children[0].value)
            if is_f1 and is_f2:
                tree.value = "sub"
                tree.children[1] = tree.children[1].children[1]
                tree.children[0].value = f1 + f2
        elif tree.value == "sub" and tree.children[1].value == "add":
            is_f1, f1 = self.is_float(tree.children[0].value)
            is_f2, f2 = self.is_float(tree.children[1].children[0].value)
            if is_f1 and is_f2:
                tree.value = "sub"
                tree.children[1] = tree.children[1].children[1]
                tree.children[0].value = f1 - f2
        elif tree.value == "sub" and tree.children[1].value == "sub":
            is_f1, f1 = self.is_float(tree.children[0].value)
            is_f2, f2 = self.is_float(tree.children[1].children[0].value)
            if is_f1 and is_f2:
                tree.children[1] = tree.children[1].children[1]
                tree.children[0].value = f1 - f2
        elif tree.value == "mul" and tree.children[1].value == "mul":
            is_f1, f1 = self.is_float(tree.children[0].value)
            is_f2, f2 = self.is_float(tree.children[1].children[0].value)
            if is_f1 and is_f2:
                tree.children[0].unit = (
                    tree.children[0].unit + tree.children[1].children[0].unit
                )
                tree.children[1] = tree.children[1].children[1]
                tree.children[0].value = f1 * f2

    def partial_reduce_const_inv(self, tree):
        for node in tree.children:
            self.partial_reduce_const_inv(node)
        if tree.value == "mul":
            is_f1, f1 = self.is_float(tree.children[0].value)
            if is_f1:
                node = tree
                while node.value == "mul":
                    node = node.children[1]
                if node.value != "inv" or node.children[0].value != "mul":
                    return
                is_f2, f2 = self.is_float(node.children[0].children[0].value)
                if is_f2:
                    tree.children[0].value = f1 / f2
                    tree.children[0].unit = (
                        tree.children[0].unit - node.children[0].children[0].unit
                    )
                    node.children[0] = node.children[0].children[1]

    def simplify_tree(self, tree):
        self.reduce_const(tree)
        swap_flag = True
        while swap_flag:
            swap_flag = self.swap_node(tree)
        self.partial_reduce_const(tree)
        self.partial_reduce_const_inv(tree)
        self._dfs_units(tree)

    def _div_const_for_node(self, node, const, inv_flag):
        if inv_flag == 1:
            node.value = str(float(node.value) * const)
        elif const != 0:
            node.value = str(float(node.value) / const)
        else:
            node.value = str(float(node.value) / 1e-10)

    def _unify_const(self, node, const_node):
        dfs_stack = [(node, 0, [])]
        while dfs_stack:
            current_node, inv_flag, additive_node_list = dfs_stack.pop(0)
            if current_node.value == "mul":
                is_f1, f1 = self.is_float(current_node.children[0].value)
                is_f2, f2 = self.is_float(current_node.children[1].value)
                if is_f1:
                    current_node.children[0].value = "1.0"
                    self._div_const_for_node(const_node, f1, (inv_flag + 1) % 2)
                    for add_node, _inv_flag in additive_node_list:
                        self._div_const_for_node(add_node, f1, _inv_flag)
                elif is_f2:
                    current_node.children[1].value = "1.0"
                    self._div_const_for_node(const_node, f2, (inv_flag + 1) % 2)
                    for add_node, _inv_flag in additive_node_list:
                        self._div_const_for_node(add_node, f2, _inv_flag)
                dfs_stack.extend(
                    [
                        (child, inv_flag, copy.deepcopy(additive_node_list))
                        for child in current_node.children
                    ]
                )
            elif current_node.value == "inv":
                inv_flag = (inv_flag + 1) % 2
                is_f1, f1 = self.is_float(current_node.children[0].value)
                if is_f1:
                    current_node.children[0].value = "1.0"
                    self._div_const_for_node(const_node, f1, (inv_flag + 1) % 2)
                    for add_node, _inv_flag in additive_node_list:
                        self._div_const_for_node(add_node, f1, _inv_flag)
                dfs_stack.extend(
                    [
                        (child, inv_flag, copy.deepcopy(additive_node_list))
                        for child in current_node.children
                    ]
                )
            elif current_node.value == "neg":
                is_f1, f1 = self.is_float(current_node.children[0].value)
                if is_f1:
                    current_node.children[0].value = "1.0"
                    self._div_const_for_node(const_node, f1, (inv_flag + 1) % 2)
                    for add_node, _inv_flag in additive_node_list:
                        self._div_const_for_node(add_node, f1, _inv_flag)
                dfs_stack.extend(
                    [
                        (child, inv_flag, copy.deepcopy(additive_node_list))
                        for child in current_node.children
                    ]
                )
            elif current_node.value in ["add", "sub"]:
                is_f1, f1 = self.is_float(current_node.children[0].value)
                is_f2, f2 = self.is_float(current_node.children[1].value)
                if is_f1:
                    additive_node_list.append((current_node.children[0], inv_flag))
                    dfs_stack.append(
                        [current_node.children[1], inv_flag, additive_node_list]
                    )
                elif is_f2:
                    additive_node_list.append((current_node.children[1], inv_flag))
                    dfs_stack.append(
                        [current_node.children[0], inv_flag, additive_node_list]
                    )
            else:
                continue

    def unify_const(self, tree):
        start_nodes = []
        const_nodes = []
        dfs_stack = [(tree, 0)]
        while dfs_stack:
            current_node, search_type = dfs_stack.pop(0)
            arity = len(current_node.children)
            op = current_node.value
            if search_type == 0:
                if op == "mul":
                    is_f1, f1 = self.is_float(current_node.children[0].value)
                    is_f2, f2 = self.is_float(current_node.children[1].value)
                    if is_f1:
                        start_nodes.append(current_node.children[1])
                        const_nodes.append(current_node.children[0])
                    elif is_f2:
                        start_nodes.append(current_node.children[0])
                        const_nodes.append(current_node.children[1])
                    if is_f1 or is_f2:
                        dfs_stack.extend((child, 1) for child in current_node.children)
                    else:
                        dfs_stack.extend((child, 0) for child in current_node.children)
                else:
                    dfs_stack.extend((child, 0) for child in current_node.children)
            elif arity == 1 and op not in [
                "pow2",
                "pow3",
                "pow4",
                "pow5",
                "inv",
                "neg",
                "sqrt",
            ]:
                dfs_stack.append((current_node.children[0], 0))
            else:
                dfs_stack.extend((child, 1) for child in current_node.children)
        for s_node, c_node in zip(start_nodes, const_nodes):
            self._unify_const(s_node, c_node)

    def unify_const2(self, tree):
        dfs_stack = [tree]
        while dfs_stack:
            current_node = dfs_stack.pop(0)
            op = current_node.value
            arity = len(current_node.children)
            if arity == 2:
                is_f1, f1 = self.is_float(
                    current_node.children[0].value, allow_int=False
                )
                is_f2, f2 = self.is_float(
                    current_node.children[1].value, allow_int=False
                )
                if op == "add":
                    v = "0"
                elif op == "mul":
                    v = "1"
                elif op == "sub":
                    v = "0"
                else:
                    dfs_stack.extend(current_node.children)
                    continue
                if is_f1:
                    current_node.children[0].value = v
                elif is_f2:
                    current_node.children[1].value = v
            dfs_stack.extend(current_node.children)

    def unify_const3(self, tree):
        dfs_stack = [(tree, False)]
        while dfs_stack:
            current_node, is_inv = dfs_stack.pop(0)
            op = current_node.value
            arity = len(current_node.children)
            if arity == 2:
                is_f1, f1 = self.is_float(
                    current_node.children[0].value, allow_int=False
                )
                is_f2, f2 = self.is_float(
                    current_node.children[1].value, allow_int=False
                )
                int_simplify = False
                if (op == "add" or op == "sub") and is_inv:
                    int_simplify = True
                else:
                    dfs_stack.extend(
                        [(child, is_inv) for child in current_node.children]
                    )
                    continue
                if is_f1 and int_simplify:
                    current_node.children[0].value = str(
                        round(float(current_node.children[0].value))
                    )
                elif is_f2 and int_simplify:
                    current_node.children[1].value = str(
                        round(float(current_node.children[1].value))
                    )
            if op == "inv":
                dfs_stack.extend(
                    [(child, not is_inv) for child in current_node.children]
                )
            else:
                dfs_stack.extend([(child, is_inv) for child in current_node.children])

    def relabel_variables(self, tree, variables):
        active_variables = []
        for elem in tree.prefix().split(","):
            if elem.startswith("x_"):
                active_variables.append(variables[int(elem[2:])])
        active_variables = list(set(active_variables))
        input_dimension = len(active_variables)
        if input_dimension == 0:
            return 0, None
        transition_dic = {
            f"x_{variables.index(v)}": f"x_{j}" for j, v in enumerate(active_variables)
        }
        self.apply_transition(tree, transition_dic)
        return tree, active_variables

    def apply_transition(self, tree, transition_dic):
        stack = [tree]
        while stack:
            temp = stack.pop(0)
            temp.value = transition_dic.get(temp.value, temp.value)
            stack += temp.children

    @timeout(5)
    def parse_sympy_expr(self, expr, simplify=False):
        expr = sp.parse_expr(expr, local_dict=self.locals)
        if simplify:
            expr = sp.simplify(expr)
        preorder_traversal = list(sp.preorder_traversal(expr))
        if sp.I in preorder_traversal:
            raise ValueError("Complex number are ilegal!")
        elif sp.zoo in preorder_traversal:
            raise ValueError("Infinite number are ilegal!")
        elif sp.nan in preorder_traversal:
            raise ValueError("NaN number are ilegal!")
        return expr

    def _is_float(self, f):
        try:
            _ = float(f)
            return True
        except:
            return False

    def function_to_skeleton(
        self,
        tree,
        skeletonize_integers=False,
        constants_with_idx=False,
        ignore=[],
        ignore_pow=False,
    ):
        units = []
        prefix = []
        constants = []
        current_const_idx = 0
        dfs_stack = [tree]
        while dfs_stack:
            node = dfs_stack.pop(0)
            pre = node.value
            unit = node.unit
            try:
                f = float(pre)
                is_float = True
                if f == int(f):
                    is_float = True
                if f in ignore:
                    is_float = False
            except ValueError:
                is_float = False
            if pre == "pow" and ignore_pow:
                node.children[-1].pow_consts = True
            if is_float and ignore_pow and hasattr(node, "pow_consts"):
                is_float = False
                del node.pow_consts
            elif hasattr(node, "pow_consts") and node.children:
                for c in node.children:
                    c.pow_consts = True
                del node.pow_consts
            elif hasattr(node, "pow_consts"):
                del node.pow_consts
            if is_float or pre is self.constants and skeletonize_integers:
                try:
                    value = float(pre)
                except:
                    value = getattr(np, pre)
                while (
                    prefix
                    and prefix[-1] in self.unaries
                    or len(prefix) >= 2
                    and (
                        self._is_float(prefix[-1]) or prefix[-1].startswith("CONSTANT")
                    )
                    and prefix[-2] in self.binary
                ):
                    if prefix[-1] in self.unaries and not (
                        prefix[-1] == "inv" and value == 0
                    ):
                        if prefix[-1] == "neg":
                            value = -value
                        elif prefix[-1] == "inv":
                            value = 1 / value
                        elif prefix[-1][:3] == "pow":
                            value = value ** int(prefix[-1][3])
                        else:
                            value = getattr(np, prefix[-1])(value)
                        unit = units[-1]
                        prefix.pop(-1)
                        units.pop(-1)
                    else:
                        if self._is_float(prefix[-1]):
                            value2 = float(prefix[-1])
                        elif prefix[-1].startswith("CONSTANT"):
                            value2 = constants[-1]
                            constants.pop(-1)
                            current_const_idx -= 1
                        if prefix[-2] == "add":
                            value = value2 + value
                        elif prefix[-2] == "sub":
                            value = value2 - value
                        elif prefix[-2] == "mul":
                            value = value2 * value
                        elif prefix[-2] == "pow":
                            value = value2**value
                        unit = units[-2]
                        prefix.pop(-1)
                        units.pop(-1)
                        prefix.pop(-1)
                        units.pop(-1)
                if constants_with_idx:
                    prefix.append("CONSTANT_{}".format(current_const_idx))
                else:
                    prefix.append("CONSTANT")
                constants.append(value)
                current_const_idx += 1
            else:
                prefix.append(str(pre))
            units.append(unit)
            dfs_stack = node.children + dfs_stack
        if units[0] is None:
            new_tree = self.equation_encoder.decode(prefix)
        else:
            new_tree = self.equation_encoder.decode(
                prefix, decode_physical_units="skeleton", units=units
            )
        return new_tree, constants

    def wrap_equation_floats(self, tree, constants):
        transition_dic = {f"CONSTANT_{j}": constants[j] for j in range(len(constants))}
        self.apply_transition(tree, transition_dic)
        return tree

    def order_datapoints(self, inputs, outputs):
        mean_input = inputs.mean(0)
        distance_to_mean = np.linalg.norm(inputs - mean_input, axis=-1)
        order_by_distance = np.argsort(distance_to_mean)
        return inputs[order_by_distance], outputs[order_by_distance]

    def generate_datapoints(
        self,
        tree,
        rng,
        n_input_points,
        n_prediction_points,
        prediction_sigmas,
        input_distribution_type,
        rotate=True,
        offset=None,
        **kwargs,
    ):
        _generate_datapoints = self._generate_datapoints
        if isinstance(input_distribution_type, str):
            input_distribution_type = [
                *self.params.generate_datapoints_distribution.split(","),
                input_distribution_type,
            ]
        inputs, outputs = _generate_datapoints(
            tree=tree,
            rng=rng,
            n_points=n_input_points,
            scale=1,
            rotate=rotate,
            offset=offset,
            input_distribution_type=input_distribution_type,
            **kwargs,
        )
        if inputs is None:
            return tree, None
        datapoints = {"fit": (inputs, outputs)}
        if n_prediction_points == 0:
            return tree, datapoints
        for sigma_factor in prediction_sigmas:
            inputs, outputs = _generate_datapoints(
                tree=tree,
                rng=rng,
                n_points=n_prediction_points,
                scale=sigma_factor,
                rotate=rotate,
                offset=offset,
                input_distribution_type=input_distribution_type,
                **kwargs,
            )
            if inputs is None:
                return tree, None
            datapoints["predict_{}".format(sigma_factor)] = inputs, outputs
        return tree, datapoints

    def _generate_datapoints(
        self,
        tree,
        n_points,
        scale,
        rng,
        input_dimension,
        input_distribution_type,
        n_centroids,
        max_trials,
        rotate=True,
        offset=None,
    ):
        inputs, outputs = [], []
        remaining_points = n_points
        trials = 0
        while remaining_points > 0 and trials < max_trials:
            if trials % 100 == 0:
                if input_distribution_type[1] == "single":
                    n_centroids = 1
                    if input_distribution_type[0] == "positive":
                        means = np.ones(shape=(n_centroids, input_dimension)) * 2
                        covariances = np.ones(shape=(n_centroids, input_dimension)) * 3
                    elif input_distribution_type[0] == "all":
                        means = np.zeros(shape=(n_centroids, input_dimension))
                        covariances = np.ones(shape=(n_centroids, input_dimension)) * 6
                    rotate = False
                elif input_distribution_type[1] == "multi":
                    means = rng.randn(n_centroids, input_dimension)
                    covariances = rng.uniform(1, 3, size=(n_centroids, input_dimension))
                if rotate:
                    rotations = [
                        (
                            special_ortho_group.rvs(input_dimension)
                            if input_dimension > 1
                            else np.identity(1)
                        )
                        for _ in range(n_centroids)
                    ]
                else:
                    rotations = [
                        np.identity(input_dimension) for _ in range(n_centroids)
                    ]
                weights = rng.uniform(0, 1, size=(n_centroids,))
                weights /= np.sum(weights)
                n_points_comp = rng.multinomial(n_points, weights)
            if input_distribution_type[2] == "gaussian":
                input = np.vstack(
                    [
                        (
                            rng.multivariate_normal(
                                mean, np.diag(covariance), int(sample)
                            )
                            @ rotation
                        )
                        for mean, covariance, rotation, sample in zip(
                            means, covariances, rotations, n_points_comp
                        )
                    ]
                )
            elif input_distribution_type[2] == "uniform":
                input = np.vstack(
                    [
                        (
                            (
                                mean
                                + rng.uniform(-1, 1, size=(sample, input_dimension))
                                * np.sqrt(covariance)
                            )
                            @ rotation
                        )
                        for mean, covariance, rotation, sample in zip(
                            means, covariances, rotations, n_points_comp
                        )
                    ]
                )
            if input_distribution_type[0] == "all":
                input = (input - np.mean(input, axis=0, keepdims=True)) / np.std(
                    input, axis=0, keepdims=True
                )
            elif input_distribution_type[0] == "positive":
                input = np.abs(input)
            input *= scale
            if offset is not None:
                mean, std = offset
                input *= std
                input += mean
            output = tree.val(input).reshape((-1, 1))
            is_nan_idx = np.any(np.isnan(output), -1)
            input = input[~is_nan_idx, :]
            output = output[~is_nan_idx, :]
            output[np.abs(output) >= self.max_number] = np.nan
            output[np.abs(output) == np.inf] = np.nan
            is_nan_idx = np.any(np.isnan(output), -1)
            input = input[~is_nan_idx, :]
            output = output[~is_nan_idx, :]
            valid_points = output.shape[0]
            trials += 1
            remaining_points -= valid_points
            if valid_points == 0:
                continue
            inputs.append(input)
            outputs.append(output)
        if remaining_points > 0:
            return None, None
        inputs = np.concatenate(inputs, 0)[:n_points]
        outputs = np.concatenate(outputs, 0)[:n_points]
        return inputs, outputs
