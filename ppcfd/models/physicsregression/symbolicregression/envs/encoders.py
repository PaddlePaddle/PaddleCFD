import math
from abc import ABC, abstractmethod

import numpy as np

from .generators import Node
from .utils import *


class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """

    def __init__(self, params):
        pass

    @abstractmethod
    def encode(self, val):
        pass

    @abstractmethod
    def decode(self, lst):
        pass


class GeneralEncoder:
    def __init__(self, params, symbols, all_operators):
        self.float_encoder = FloatSequences(params)
        self.equation_encoder = Equation(
            params, symbols, self.float_encoder, all_operators
        )


class FloatSequences(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.float_precision = params.float_precision
        self.mantissa_len = params.mantissa_len
        self.max_exponent = params.max_exponent
        self.base = (self.float_precision + 1) // self.mantissa_len
        self.max_token = 10**self.base
        self.symbols = ["+", "-"]
        self.symbols.extend(
            [("N" + f"%0{self.base}d" % i) for i in range(self.max_token)]
        )
        self.symbols.extend(
            [("E" + str(i)) for i in range(-self.max_exponent, self.max_exponent + 1)]
        )

    def encode(self, values):
        """
        Write a float number
        """
        precision = self.float_precision
        if len(values.shape) == 1:
            seq = []
            value = values
            for val in value:
                assert val not in [-np.inf, np.inf]
                sign = "+" if val >= 0 else "-"
                m, e = (f"%.{precision}e" % val).split("e")
                i, f = m.lstrip("-").split(".")
                i = i + f
                tokens = chunks(i, self.base)
                expon = int(e) - precision
                if expon < -self.max_exponent:
                    tokens = ["0" * self.base] * self.mantissa_len
                    expon = int(0)
                seq.extend(
                    [sign, *[("N" + token) for token in tokens], "E" + str(expon)]
                )
            return seq
        else:
            seqs = [self.encode(values[0])]
            N = values.shape[0]
            for n in range(1, N):
                seqs += [self.encode(values[n])]
        return seqs

    def encode_batch(self, values_batch):
        """
        批量编码NumPy数组 - 向量化优化版本

        Args:
            values_batch: shape (N,) 或 (N, D) 的NumPy数组

        Returns:
            编码后的token列表,每个元素对应一个数值
        """
        precision = self.float_precision

        # 确保是2D数组
        if len(values_batch.shape) == 1:
            values_batch = values_batch.reshape(-1, 1)

        N, D = values_batch.shape
        result = []

        # 向量化符号提取
        signs = np.where(values_batch >= 0, "+", "-")

        # 展平数组以便批量处理
        flat_values = values_batch.flatten()
        flat_signs = signs.flatten()

        # 批量格式化 - 使用列表推导式而不是循环
        formatted_strs = [f"%.{precision}e" % val for val in flat_values]

        # 批量解析
        for idx, (val, sign, fmt_str) in enumerate(zip(flat_values, flat_signs, formatted_strs)):
            assert val not in [-np.inf, np.inf]

            m, e = fmt_str.split("e")
            i_part, f_part = m.lstrip("-").split(".")
            mantissa = i_part + f_part
            tokens = chunks(mantissa, self.base)
            expon = int(e) - precision

            if expon < -self.max_exponent:
                tokens = ["0" * self.base] * self.mantissa_len
                expon = 0

            result.append(
                [sign, *[("N" + token) for token in tokens], "E" + str(expon)]
            )

        return result

    def decode(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) == 0:
            return None
        seq = []
        for val in chunks(lst, 2 + self.mantissa_len):
            for x in val:
                if x[0] not in ["-", "+", "E", "N"]:
                    return np.nan
            try:
                sign = 1 if val[0] == "+" else -1
                mant = ""
                for x in val[1:-1]:
                    mant += x[1:]
                mant = int(mant)
                exp = int(val[-1][1:])
                value = sign * mant * 10**exp
                value = float(value)
            except Exception:
                value = np.nan
            seq.append(value)
        return seq


class Equation(Encoder):
    def __init__(self, params, symbols, float_encoder, all_operators):
        super().__init__(params)
        self.params = params
        self.max_int = self.params.max_int
        self.symbols = symbols
        if params.extra_unary_operators != "":
            self.extra_unary_operators = self.params.extra_unary_operators.split(",")
        else:
            self.extra_unary_operators = []
        if params.extra_binary_operators != "":
            self.extra_binary_operators = self.params.extra_binary_operators.split(",")
        else:
            self.extra_binary_operators = []
        self.float_encoder = float_encoder
        self.all_operators = all_operators

    def encode(self, tree, zero_physical_units=False):
        res = []
        units = []
        if self.params.decode_physical_units is None:
            for elem in tree.prefix().split(","):
                try:
                    val = float(elem)
                    if val == int(val) and abs(val) < 10:
                        res.extend(self.write_int(int(val)))
                    else:
                        res.extend(self.float_encoder.encode(np.array([val])))
                except ValueError:
                    res.append(elem)
            return res, units
        elif self.params.decode_physical_units == "single-seq":
            stack = [tree]
            while stack:
                temp = stack.pop(0)
                elem = temp.value
                try:
                    val = float(elem)
                    if val == int(val) and abs(val) < 10:
                        res.extend(self.write_int(int(val)))
                    else:
                        res.extend(self.float_encoder.encode(np.array([val])))
                except ValueError:
                    res.append(elem)
                res.extend(self.units_encode(temp.unit, zero_physical_units))
                stack = temp.children + stack
            return res, units
        elif self.params.decode_physical_units == "double-seq":
            stack = [tree]
            while stack:
                temp = stack.pop(0)
                elem = temp.value
                try:
                    val = float(elem)
                    if val == int(val) and abs(val) < 10:
                        res.extend(self.write_int(int(val)))
                    else:
                        res.extend(self.float_encoder.encode(np.array([val])))
                        units.append(self.units_encode(temp.unit, zero_physical_units))
                        units.append(self.units_encode(temp.unit, zero_physical_units))
                except ValueError:
                    res.append(elem)
                units.append(self.units_encode(temp.unit, zero_physical_units))
                stack = temp.children + stack
            return res, units

    def split_at_value(self, lst, value):
        indices = [i for i, x in enumerate(lst) if x == value]
        res = []
        for start, end in zip(
            [0, *[(i + 1) for i in indices]], [*[(i - 1) for i in indices], len(lst)]
        ):
            res.append(lst[start : end + 1])
        return res

    def _decode(self, lst):
        if len(lst) == 0:
            return None, 0
        elif "OOD" in lst[0]:
            return None, 0
        elif lst[0].lower() in self.all_operators.keys():
            res = Node(lst[0].lower(), self.params)
            arity = self.all_operators[lst[0].lower()]
            pos = 1
            for i in range(arity):
                child, length = self._decode(lst[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        elif lst[0].startswith("INT"):
            val, length = self.parse_int(lst)
            node = Node(str(val), self.params)
            return node, length
        elif lst[0] == "+" or lst[0] == "-":
            try:
                val = self.float_encoder.decode(lst[:3])[0]
                assert not np.isnan(val)
            except Exception as e:
                return None, 0
            node = Node(str(val), self.params)
            return node, 3
        elif lst[0].startswith("CONSTANT") or lst[0] == "y":
            node = Node(lst[0], self.params)
            return node, 1
        elif lst[0] in self.symbols:
            node = Node(lst[0], self.params)
            return node, 1
        elif lst[0] == "empty":
            node = Node(lst[0], self.params)
            return node, 1
        else:
            try:
                float(lst[0])
                node = Node(lst[0], self.params)
                return node, 1
            except:
                return None, 0

    def _decode_with_units1(self, lst):
        if len(lst) == 0:
            return None, 0
        pos = 0
        if "OOD" in lst[pos]:
            return None, 0
        elif lst[pos] in self.all_operators.keys():
            res = Node(lst[pos], self.params)
            arity = self.all_operators[lst[pos]]
            try:
                unit = self.units_decode(lst[pos + 1 : pos + 1 + 5])
            except:
                return None, 0
            res.unit = unit
            pos += 1 + 5
            for i in range(arity):
                child, length = self._decode_with_units1(lst[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        elif lst[pos].startswith("INT"):
            val, length = self.parse_int(lst[pos:])
            node = Node(str(val), self.params)
            try:
                unit = self.units_decode(lst[pos + length : pos + length + 5])
            except:
                return None, 0
            node.unit = unit
            return node, length + pos + 5
        elif lst[pos] == "+" or lst[pos] == "-":
            try:
                val = self.float_encoder.decode(lst[pos : pos + 3])[0]
                assert not np.isnan(val)
            except Exception as e:
                return None, 0
            node = Node(str(val), self.params)
            try:
                unit = self.units_decode(lst[pos + 3 : pos + 3 + 5])
            except:
                return None, 0
            node.unit = unit
            return node, 3 + pos + 5
        elif lst[pos].startswith("CONSTANT") or lst[pos] == "y":
            node = Node(lst[pos], self.params)
            try:
                unit = self.units_decode(lst[pos + 1 : pos + 1 + 5])
            except:
                return None, 0
            node.unit = unit
            return node, 1 + pos + 5
        elif lst[pos] in self.symbols:
            node = Node(lst[pos], self.params)
            try:
                unit = self.units_decode(lst[pos + 1 : pos + 1 + 5])
            except:
                return None, 0
            node.unit = unit
            return node, 1 + pos + 5
        else:
            try:
                units = self.units_decode(lst[pos + 1 : pos + 1 + 5])
            except:
                return None, 0
            try:
                float(lst[pos])
                node = Node(lst[pos], self.params)
                node.unit = units
                return node, 1 + pos + 5
            except:
                return None, 0

    def _decode_with_units2(self, lst, units):
        if len(lst) == 0:
            return None, 0
        pos = 0
        try:
            unit = self.units_decode(units[pos : pos + 5])
        except:
            return None, 0
        if "OOD" in lst[pos]:
            return None, 0
        elif lst[pos] in self.all_operators.keys():
            res = Node(lst[pos], self.params)
            arity = self.all_operators[lst[pos]]
            res.unit = unit
            pos += 1
            for i in range(arity):
                child, length = self._decode_with_units2(lst[pos:], units[pos * 5 :])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        elif lst[pos].startswith("INT"):
            val, length = self.parse_int(lst[pos:])
            node = Node(str(val), self.params)
            node.unit = unit
            return node, length + pos
        elif lst[pos] == "+" or lst[pos] == "-":
            try:
                val = self.float_encoder.decode(lst[pos : pos + 3])[0]
                assert not np.isnan(val)
            except Exception as e:
                return None, 0
            node = Node(str(val), self.params)
            node.unit = unit
            return node, 3 + pos
        elif lst[pos].startswith("CONSTANT") or lst[pos] == "y":
            node = Node(lst[pos], self.params)
            node.unit = unit
            return node, 1 + pos
        elif lst[pos] in self.symbols:
            node = Node(lst[pos], self.params)
            node.unit = unit
            return node, 1 + pos
        else:
            try:
                float(lst[pos])
                node = Node(lst[pos], self.params)
                node.unit = unit
                return node, 1 + pos
            except:
                return None, 0

    def _decode_with_units3(self, lst, units):
        if len(lst) == 0:
            return None, 0
        pos = 0
        unit = units[pos]
        if lst[pos] in self.all_operators.keys():
            res = Node(lst[pos], self.params)
            arity = self.all_operators[lst[pos]]
            res.unit = unit
            pos += 1
            for i in range(arity):
                child, length = self._decode_with_units3(lst[pos:], units[pos:])
                if child is None:
                    return None, pos
                res.push_child(child)
                pos += length
            return res, pos
        elif lst[pos].startswith("CONSTANT") or lst[pos] == "y":
            node = Node(lst[pos], self.params)
            node.unit = unit
            return node, 1 + pos
        elif lst[pos] in self.symbols:
            node = Node(lst[pos], self.params)
            node.unit = unit
            return node, 1 + pos
        else:
            return None, 0

    def check_units(self, tree, xy_units):
        stack = [tree]
        while stack:
            temp = stack.pop(0)
            value = str(temp.value)
            if value in ["add", "sub"]:
                if any(temp.children[0].unit != temp.children[1].unit):
                    return False
            elif value in ["mul"]:
                if any(temp.unit != temp.children[0].unit + temp.children[1].unit):
                    return False
            elif value in ["div"]:
                if any(temp.unit - temp.children[1].unit != temp.children[0].unit):
                    return False
            elif value in ["abs", "neg"]:
                if any(temp.unit != temp.children[0].unit):
                    return False
            elif value in ["inv"]:
                if any(-temp.unit != temp.children[0].unit):
                    return False
            elif value in ["sqrt"]:
                if any(temp.unit * 2 != temp.children[0].unit):
                    return False
            elif value in ["pow2", "pow3", "pow4", "pow5"]:
                if any(temp.unit != temp.children[0].unit * int(temp.value[-1])):
                    return False
            elif value in self.all_operators:
                if any(temp.unit != 0):
                    return False
            elif value.startswith("x_"):
                idx = int(temp.value[2:])
                if not idx < len(xy_units) - 1:
                    return False
                dim = xy_units[idx]
                # 修复：确保dim是np.ndarray类型，避免PaddlePaddle的类型提升错误
                if not isinstance(dim, np.ndarray):
                    dim = np.array(dim, dtype=np.float64)
                if any(temp.unit != dim):
                    return False
            stack += temp.children
        if isinstance(xy_units[-1], str) and xy_units[-1] == "<UNKNOWN_PHYSICAL_UNITS>":
            return True
        # 修复：确保xy_units[-1]是np.ndarray类型
        last_unit = xy_units[-1]
        if not isinstance(last_unit, np.ndarray):
            last_unit = np.array(last_unit, dtype=np.float64)
        if any(tree.unit != last_unit):
            return False
        return True

    def decode(
        self,
        lst,
        xy_units=None,
        decode_physical_units=None,
        units=None,
        lable_unit_fn=None,
    ):
        trees = []
        lists = self.split_at_value(lst, "|")
        for lst in lists:
            if decode_physical_units is None:
                tree = self._decode(lst)[0]
                if tree is not None and xy_units is not None:
                    try:
                        lable_unit_fn(tree, units=xy_units)
                    except:
                        return None
            elif decode_physical_units == "single-seq":
                tree = self._decode_with_units1(lst)[0]
            elif decode_physical_units == "double-seq":
                tree = self._decode_with_units2(lst, units)[0]
            elif decode_physical_units == "skeleton":
                tree = self._decode_with_units3(lst, units)[0]
            else:
                raise ValueError()
            if tree is None:
                return None
            if xy_units is not None and not self.check_units(tree, xy_units):
                if self.params.units_criterion_constrain:
                    return None
                pass
            trees.append(tree)
        assert len(trees) == 1
        return trees[0]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.max_int
        val = 0
        i = 0
        for x in lst[1:]:
            if not x.rstrip("-").isdigit():
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == "INT-":
            val = -val
        return val, i + 1

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        """
        if not self.params.use_sympy:
            return [str(val)]
        base = self.max_int
        res = []
        max_digit = abs(base)
        neg = val < 0
        val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        res.append("INT-" if neg else "INT+")
        return res[::-1]

    def units_encode(self, units, zero_physical_units=False):
        if isinstance(units, str):
            return [units]
        if zero_physical_units:
            units = np.array([0, 0, 0, 0, 0])
        m, s, kg, T, V = list(map(lambda x: str(int(x)), units))
        m, s, kg, T, V = list(
            variable + num
            for variable, num in zip(["M", "S", "K", "T", "V"], [m, s, kg, T, V])
        )
        if self.params.dim_length == 2:
            return [m + s, kg + T + V]
        elif self.params.dim_length == 5:
            return [m, s, kg, T, V]
        else:
            raise ValueError(f"Error dim lenght: {self.params.dim_length}")

    def units_decode(self, lst):
        assert len(lst) == self.params.dim_length
        if self.params.dim_length == 2:
            m, s = list(map(int, lst[0][1:].split("S")))
            kg, t = lst[1][1:].split("T")
            kg, t, v = list(map(int, [kg] + t.split("V")))
        elif self.params.dim_length == 5:
            m, s, kg, t, v = list(map(lambda x: int(x[1:]), lst))
        else:
            raise ValueError(f"Error dim lenght: {self.params.dim_length}")
        lst = np.array([m, s, kg, t, v])
        return lst
