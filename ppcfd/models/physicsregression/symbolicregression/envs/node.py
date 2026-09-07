import numpy as np
import scipy

from .operators import binary_complex_dic
from .operators import math_constants
from .operators import unary_complex_dic


class Node:
    def __init__(self, value, params, children=None):
        self.children = children if children else []
        self.params = params
        self.unit = None
        self.update_value(value)

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += "," + c.prefix()
        return s

    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self):
        nb_children = len(self.children)
        if nb_children == 0:
            if str(self.value).lstrip("-").isdigit():
                return str(self.value)
            else:
                s = str(self.value)
                return s
        if nb_children == 1:
            s = str(self.value)
            if s == "pow2":
                s = "(" + self.children[0].infix() + ")**2"
            elif s == "pow3":
                s = "(" + self.children[0].infix() + ")**3"
            elif s == "pow4":
                s = "(" + self.children[0].infix() + ")**4"
            elif s == "pow5":
                s = "(" + self.children[0].infix() + ")**5"
            elif s == "neg":
                s = "(-" + self.children[0].infix() + ")"
            else:
                s = s + "(" + self.children[0].infix() + ")"
            return s
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, x, deterministic=True):
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return x[:, dim].copy()
            elif str(self.value) == "rand":
                if deterministic:
                    return np.zeros((x.shape[0],))
                return np.random.randn(x.shape[0])
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value).lower()) * np.ones((x.shape[0],))
            else:
                return float(self.value) * np.ones((x.shape[0],))
        if self.value == "add":
            return self.children[0].val(x) + self.children[1].val(x)
        if self.value == "sub":
            return self.children[0].val(x) - self.children[1].val(x)
        if self.value == "neg":
            return -self.children[0].val(x)
        if self.value == "mul":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)
            try:
                return m1 * m2
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow":
            m1, m2 = self.children[0].val(x), self.children[1].val(x)
            try:
                return np.power(m1, m2)
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "max":
            return np.maximum(self.children[0].val(x), self.children[1].val(x))
        if self.value == "min":
            return np.minimum(self.children[0].val(x), self.children[1].val(x))
        if self.value == "div":
            denominator = self.children[1].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return self.children[0].val(x) / denominator
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "inv":
            denominator = self.children[0].val(x)
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "log":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator <= 0.0] = np.nan
            try:
                return np.log(numerator)
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "sqrt":
            numerator = self.children[0].val(x)
            if self.params.use_abs:
                numerator[numerator <= 0.0] *= -1
            else:
                numerator[numerator < 0.0] = np.nan
            try:
                return np.sqrt(numerator)
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow2":
            numerator = self.children[0].val(x)
            try:
                return numerator**2
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow3":
            numerator = self.children[0].val(x)
            try:
                return numerator**3
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow4":
            numerator = self.children[0].val(x)
            try:
                return numerator**4
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "pow5":
            numerator = self.children[0].val(x)
            try:
                return numerator**5
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "exp":
            numerator = self.children[0].val(x)
            numerator[numerator > 700] = np.inf
            numerator[numerator < -700] = -np.inf
            try:
                return np.exp(numerator)
            except Exception as e:
                print(e)
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        if self.value == "abs":
            return np.abs(self.children[0].val(x))
        if self.value == "sign":
            return (self.children[0].val(x) >= 0) * 2.0 - 1.0
        if self.value == "step":
            x = self.children[0].val(x)
            return x if x > 0 else 0
        if self.value == "id":
            return self.children[0].val(x)
        if self.value == "fresnel":
            return scipy.special.fresnel(self.children[0].val(x))[0]
        if self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(n, self.children[0].val(x))[
                0
            ]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(self.children[0].val(x))
                except Exception as e:
                    nans = np.empty((x.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(self.children[0].val(x))
            assert False, f"Could not find function: {self.value}"

    def get_recurrence_degree(self):
        recurrence_degree = 0
        if len(self.children) == 0:
            if str(self.value).startswith("x_"):
                _, _, offset = self.value.split("_")
                offset = int(offset)
                if offset > recurrence_degree:
                    recurrence_degree = offset
            return recurrence_degree
        return max([child.get_recurrence_degree() for child in self.children])

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)

    def copy(self):
        new_node = Node(self.value, self.params)
        new_node.children = [child.copy() for child in self.children]
        new_node.unit = self.unit
        return new_node

    def push_children(self, child):
        """push a child node into self.children"""
        if isinstance(child, Node):
            self.children.append(child)
        else:
            self.children.append(Node(child, self.params))

    def update_value(self, value):
        self.value = value
        if value in unary_complex_dic:
            self.degree = 1
        elif value in binary_complex_dic:
            self.degree = 2
        else:
            self.degree = 0
        if "x_" in str(value):
            self.is_variable = True
        else:
            self.is_variable = False
        try:
            _ = float(value)
            self.is_const = True
        except:
            self.is_const = False
