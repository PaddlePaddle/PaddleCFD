"""
Symbolic Encoders.

Source: https://github.com/facebookresearch/symbolicregression/blob/main/symbolicregression/envs/encoders.py
"""
from abc import ABC, abstractmethod

import numpy as np

from ppcfd.models.prose_fd.symbol_utils.node_utils import Node, NodeList


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
        self.float_encoder = float_encoder
        self.all_operators = all_operators

    def encode(self, tree):
        """
        Encoding the tree into a list of symbols
        """
        res = []
        for elem in tree.prefix().split(","):
            try:
                val = float(elem)
                if elem.lstrip("-").isdigit():
                    res.extend(self.write_int(int(elem)))
                else:
                    res.extend(self.float_encoder.encode(np.array([val])))
            except ValueError:
                res.append(elem)
        return res

    def encode_with_placeholder(self, tree):
        """
        Encoding the tree into a list of symbols, replacing numerical values with placeholder token
        """
        res = []
        for elem in tree.prefix().split(","):
            try:
                val = float(elem)
                res.append("<PLACEHOLDER>")
            except ValueError:
                res.append(elem)
        return res

    def encode_with_noise(self, tree, sigma=0.05):
        """
        Encoding the tree into a list of symbols, adding noise to numerical values
        """
        res = []
        for elem in tree.prefix().split(","):
            try:
                val = float(elem)
                if elem.lstrip("-").isdigit():
                    val = int(elem)
                    elem = val + sigma * np.abs(val) * np.random.randn()
                    res.extend(self.write_int(int(elem)))
                else:
                    val = val + sigma * np.abs(val) * np.random.randn()
                    res.extend(self.float_encoder.encode(np.array([val])))
            except ValueError:
                res.append(elem)
        return res

    def _decode(self, lst):
        """
        Decode list of symbols in prefix notation into a tree
        """
        if len(lst) == 0:
            return None, 0
        elif "OOD" in lst[0]:
            return None, 0
        elif lst[0] in self.all_operators.keys():
            res = Node(lst[0], self.params)
            arity = self.all_operators[lst[0]]
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
            return Node(str(val), self.params), length
        elif lst[0] == "+" or lst[0] == "-":
            try:
                val = self.float_encoder.decode(lst[:3])[0]
            except Exception as e:
                return None, 0
            return Node(str(val), self.params), 3
        elif lst[0].startswith("CONSTANT") or lst[0] == "y":
            return Node(lst[0], self.params), 1
        elif lst[0] in self.symbols:
            return Node(lst[0], self.params), 1
        else:
            try:
                float(lst[0])
                return Node(lst[0], self.params), 1
            except:
                return None, 0

    def split_at_value(self, lst, value):
        indices = [i for i, x in enumerate(lst) if x == value]
        res = []
        for start, end in zip(
            [0, *[(i + 1) for i in indices]], [*[(i - 1) for i in indices], len(lst)]
        ):
            res.append(lst[start : end + 1])
        return res

    def decode(self, lst):
        """
        Decode list for multi-dimension function (split with "|") in prefix notation into NodeList trees
        """
        trees = []
        lists = self.split_at_value(lst, "|")
        for lst in lists:
            tree = self._decode(lst)[0]
            if tree is None:
                return None
            trees.append(tree)
        tree = NodeList(trees)
        return tree

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
