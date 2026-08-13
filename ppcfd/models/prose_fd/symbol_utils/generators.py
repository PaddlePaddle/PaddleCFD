import copy
from logging import getLogger

import numpy as np

from ppcfd.models.prose_fd.symbol_utils import encoders
from ppcfd.models.prose_fd.symbol_utils.node_utils import Node, NodeList, math_constants
logger = getLogger()
operators_real = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "neg": 1,
    "inv": 1,
    "sin": 1,
    "cos": 1,
    "pow": 2,
    "pow2": 1,
    "pow3": 1,
    "D": 1,
    "D2": 1,
    "diverg": 1,
    "dt": 1,
    "dx": 1,
    "dy": 1,
    "dz": 1,
}
operators_extra = dict()
all_operators = {**operators_real, **operators_extra}


class RandomFunctions:
    def __init__(self, params, special_words):
        self.params = params
        self.max_int = params.max_int
        self.max_input_dimension = params.max_input_dimension
        self.max_output_dimension = params.max_output_dimension
        self.operators = copy.deepcopy(operators_real)
        self.unaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 1
        ]
        self.binaries = [
            o for o in self.operators.keys() if np.abs(self.operators[o]) == 2
        ]
        self.constants = [
            str(i) for i in range(-self.max_int, self.max_int + 1) if i != 0
        ]
        self.constants += math_constants
        self.variables = (
            ["rand"]
            + [
                "u",
                "U",
                "v",
                "V",
                "h",
                "x",
                "X",
                "b",
                "rho",
                "p",
                "eta",
                "f",
                "F",
                "c",
                "zeta",
                "sigma",
                "w",
            ]
            + [f"u_{i}" for i in range(self.max_output_dimension)]
            + [f"v_{i}" for i in range(self.max_output_dimension)]
        )
        self.symbols = (
            list(self.operators)
            + self.constants
            + self.variables
            + ["|", "--", "INT+", "INT-", "FLOAT+", "FLOAT-", "pow", "0"]
            + ["periodic", "dirichlet", "neumann", "dirichlet_irreg"]
        )
        self.constants.remove("CONSTANT")
        self.general_encoder = encoders.GeneralEncoder(
            params, self.symbols, all_operators
        )
        self.float_encoder = self.general_encoder.float_encoder
        self.float_words = special_words + sorted(list(set(self.float_encoder.symbols)))
        self.equation_encoder = self.general_encoder.equation_encoder
        self.equation_words = sorted(list(set(self.symbols)))
        self.equation_words = special_words + self.equation_words

    def refine_floats(self, lst):
        """
        Refine floats to specified precision
        """
        return np.array(self.float_encoder.decode(self.float_encoder.encode(lst)))

    def mul_terms(self, lst):
        """
        Generate a tree containing multiplication of terms in lst
        """
        tree = None
        for i in reversed(range(len(lst))):
            cur_term = lst[i]
            if tree is None:
                if isinstance(cur_term, Node):
                    tree = cur_term
                else:
                    tree = Node(cur_term)
            elif isinstance(cur_term, Node):
                tree = Node("mul", [cur_term, tree])
            else:
                tree = Node("mul", [Node(cur_term), tree])
        return tree

    def add_terms(self, lst):
        """
        Generate a tree containing addition of terms in lst
        """
        tree = None
        for i in reversed(range(len(lst))):
            cur_term = lst[i]
            if tree is None:
                if isinstance(cur_term, Node):
                    tree = cur_term
                else:
                    tree = Node(cur_term)
            elif isinstance(cur_term, Node):
                tree = Node("add", [cur_term, tree])
            else:
                tree = Node("add", [Node(cur_term), tree])
        return tree

    def get_tree(self, tree_type, coeffs=None):
        if tree_type == "react_diff":
            raise NotImplementedError(f"{tree_type} not implemented")
        elif tree_type == "shallow_water":
            return self.tree_shallow_water(coeffs)
        elif tree_type == "incom_ns":
            return self.tree_incom_ns(coeffs)
        elif tree_type == "com_ns":
            return self.tree_com_ns(coeffs)
        elif tree_type == "incom_ns_arena":
            return self.tree_incom_ns_arena(coeffs)
        elif tree_type == "incom_ns_arena_u":
            return self.tree_incom_ns_arena_u(coeffs)
        elif tree_type == "cfdbench":
            return self.tree_cfdbench(coeffs)
        else:
            raise ValueError(f"Unknown tree type {tree_type}")

    def tree_shallow_water(self, coeffs=None):
        p = self.params
        h = Node("h")
        U = Node("U")
        eqn1 = Node("add", [Node("dt", [h]), Node("diverg", [Node("mul", [h, U])])])
        eqn2 = Node(
            "add",
            [
                Node(
                    "add",
                    [
                        Node("dt", [Node("mul", [h, U])]),
                        Node(
                            "diverg",
                            [
                                Node(
                                    "add",
                                    [
                                        self.mul_terms(["0.5", h, U, U]),
                                        self.mul_terms(["0.5", "g", h, h]),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                Node("mul", [Node("mul", [Node("g"), h]), Node("D", [Node("b")])]),
            ],
        )
        tree = NodeList([eqn1, eqn2])
        return tree

    def tree_incom_ns(self, coeffs=None):
        rho = Node("rho")
        eta = Node("eta")
        U = Node("U")
        p = Node("p")
        F = Node("F")
        c = Node("c")
        eqn1 = self.add_terms(
            [
                Node(
                    "mul",
                    [
                        rho,
                        Node(
                            "add", [Node("dt", [U]), Node("mul", [U, Node("D", [U])])]
                        ),
                    ],
                ),
                Node("D", [p]),
                Node("mul", [eta, Node("D2", [U])]),
                F,
            ]
        )
        eqn2 = Node("diverg", [U])
        eqn3 = Node("add", [Node("dt", [c]), Node("mul", [U, Node("D", [c])])])
        tree = NodeList([eqn1, eqn2, eqn3], {"U": "dirichlet"})
        return tree

    def tree_com_ns(self, coeffs=None):
        rho = Node("rho")
        U = Node("U")
        v = Node("v")
        p = Node("p")
        F = Node("F")
        sigma = Node("sigma")
        if coeffs is not None and "eta" in coeffs:
            eta = Node(str(coeffs["eta"]))
        else:
            eta = Node("eta")
        if coeffs is not None and "zeta" in coeffs:
            zeta = Node(str(coeffs["zeta"]))
        else:
            zeta = Node("zeta")
        eqn1 = self.add_terms(
            [
                Node(
                    "mul",
                    [
                        rho,
                        Node(
                            "add", [Node("dt", [U]), Node("mul", [U, Node("D", [U])])]
                        ),
                    ],
                ),
                Node("D", [p]),
                Node("mul", [eta, Node("D2", [U])]),
                Node(
                    "mul",
                    [
                        Node("add", [zeta, Node("div", [eta, Node("3")])]),
                        Node("D", [Node("diverg", [U])]),
                    ],
                ),
            ]
        )
        eqn2 = Node("add", [Node("dt", [rho]), Node("diverg", [Node("mul", [rho, U])])])
        eqn3 = Node(
            "add",
            [
                Node(
                    "dt",
                    [
                        Node(
                            "add",
                            [
                                Node("mul", [Node("1.5"), p]),
                                self.mul_terms(["0.5", rho, v, v]),
                            ],
                        )
                    ],
                ),
                Node(
                    "diverg",
                    [
                        Node(
                            "sub",
                            [
                                Node(
                                    "mul",
                                    [
                                        Node(
                                            "add",
                                            [
                                                Node("mul", [Node("2.5"), p]),
                                                self.mul_terms(["0.5", rho, v, v]),
                                            ],
                                        ),
                                        U,
                                    ],
                                ),
                                Node("mul", [U, sigma]),
                            ],
                        )
                    ],
                ),
            ],
        )
        tree = NodeList([eqn1, eqn2, eqn3], {"U": "periodic"})
        return tree

    def tree_incom_ns_arena(self, coeffs=None):
        rho = Node("rho")
        eta = Node("eta")
        U = Node("U")
        p = Node("p")
        c = Node("c")
        if coeffs is not None and "F" in coeffs:
            F = Node(str(coeffs["F"]))
        else:
            F = Node("F")
        eqn1 = self.add_terms(
            [
                Node(
                    "mul",
                    [
                        rho,
                        Node(
                            "add", [Node("dt", [U]), Node("mul", [U, Node("D", [U])])]
                        ),
                    ],
                ),
                Node("D", [p]),
                Node("mul", [eta, Node("D2", [U])]),
                F,
            ]
        )
        eqn2 = Node("diverg", [U])
        eqn3 = Node("add", [Node("dt", [c]), Node("mul", [U, Node("D", [c])])])
        tree = NodeList([eqn1, eqn2, eqn3], {"U": "dirichlet", "c": "neumann"})
        return tree

    def tree_incom_ns_arena_u(self, coeffs=None):
        return self.tree_incom_ns_arena(coeffs={"F": "0.5"})

    def tree_cfdbench(self, coeffs=None):
        rho = Node("rho")
        eta = Node("eta")
        U = Node("U")
        p = Node("p")
        eqn1 = self.add_terms(
            [
                Node(
                    "mul",
                    [
                        rho,
                        Node(
                            "add", [Node("dt", [U]), Node("mul", [U, Node("D", [U])])]
                        ),
                    ],
                ),
                Node("D", [p]),
                Node("mul", [eta, Node("D2", [U])]),
            ]
        )
        eqn2 = Node("diverg", [U])
        tree = NodeList([eqn1, eqn2], {"U": "dirichlet_irreg"})
        return tree
