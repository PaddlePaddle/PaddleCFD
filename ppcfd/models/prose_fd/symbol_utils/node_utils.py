import numpy as np
import scipy.special

math_constants = ["CONSTANT", "g"]


class Node:
    def __init__(self, value, children=None, params=None):
        self.value = value
        self.children = children if children else []
        self.params = params

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
            if self.value.lstrip("-").isdigit():
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

    def val(self, u, deterministic=True, space_dim=0):
        if len(self.children) == 0:
            if space_dim == 0:
                if str(self.value).startswith("u_"):
                    _, dim = self.value.split("_")
                    dim = int(dim)
                    return u[:, dim, 0]
                elif str(self.value) in math_constants:
                    return getattr(np, str(self.value)) * np.ones((u.shape[0],))
                else:
                    return float(self.value) * np.ones((u.shape[0],))
            elif str(self.value).startswith("u_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[dim, :, :, 0]
            elif str(self.value).startswith("ut_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[dim, :, :, 1]
            elif str(self.value).startswith("utt_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[dim, :, :, 2]
            elif str(self.value).startswith("ux_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[dim, :, :, 3]
            elif str(self.value).startswith("uxx_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[dim, :, :, 4]
            elif str(self.value).startswith("uxxx_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[dim, :, :, 5]
            elif str(self.value).startswith("uxxxx_"):
                _, dim = self.value.split("_")
                dim = int(dim)
                return u[dim, :, :, 6]
            elif str(self.value) == "x":
                return u[0, :, :, 7]
            elif str(self.value) in math_constants:
                return getattr(np, str(self.value)) * np.ones((u.shape[1], u.shape[2]))
            else:
                return float(self.value) * np.ones((u.shape[1], u.shape[2]))
        elif self.value == "add":
            return self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            ) + self.children[1].val(
                u, deterministic=deterministic, space_dim=space_dim
            )
        elif self.value == "sub":
            return self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            ) - self.children[1].val(
                u, deterministic=deterministic, space_dim=space_dim
            )
        elif self.value == "neg":
            return -self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            )
        elif self.value == "mul":
            m1, m2 = self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            ), self.children[1].val(u, deterministic=deterministic, space_dim=space_dim)
            try:
                return m1 * m2
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        elif self.value == "pow":
            m1, m2 = self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            ), self.children[1].val(u, deterministic=deterministic, space_dim=space_dim)
            try:
                return np.power(m1, m2)
            except Exception as e:
                nans = np.empty((m1.shape[0],))
                nans[:] = np.nan
                return nans
        elif self.value == "div":
            denominator = self.children[1].val(
                u, deterministic=deterministic, space_dim=space_dim
            )
            denominator[denominator == 0.0] = np.nan
            try:
                return (
                    self.children[0].val(
                        u, deterministic=deterministic, space_dim=space_dim
                    )
                    / denominator
                )
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        elif self.value == "inv":
            denominator = self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            )
            denominator[denominator == 0.0] = np.nan
            try:
                return 1 / denominator
            except Exception as e:
                nans = np.empty((denominator.shape[0],))
                nans[:] = np.nan
                return nans
        elif self.value == "pow2":
            numerator = self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            )
            try:
                return numerator**2
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        elif self.value == "pow3":
            numerator = self.children[0].val(
                u, deterministic=deterministic, space_dim=space_dim
            )
            try:
                return numerator**3
            except Exception as e:
                nans = np.empty((numerator.shape[0],))
                nans[:] = np.nan
                return nans
        elif self.value.startswith("eval"):
            n = self.value[-1]
            return getattr(scipy.special, self.value[:-1])(
                n,
                self.children[0].val(
                    u, deterministic=deterministic, space_dim=space_dim
                ),
            )[0]
        else:
            fn = getattr(np, self.value, None)
            if fn is not None:
                try:
                    return fn(
                        self.children[0].val(
                            u, deterministic=deterministic, space_dim=space_dim
                        )
                    )
                except Exception as e:
                    nans = np.empty((u.shape[0],))
                    nans[:] = np.nan
                    return nans
            fn = getattr(scipy.special, self.value, None)
            if fn is not None:
                return fn(
                    self.children[0].val(
                        u, deterministic=deterministic, space_dim=space_dim
                    )
                )
            assert False, "Could not find function: {}".format(self.value)

    def replace_node_value(self, old_value, new_value):
        if self.value == old_value:
            self.value = new_value
        for child in self.children:
            child.replace_node_value(old_value, new_value)


class NodeList:
    def __init__(self, nodes, BCs=None):
        self.nodes = []
        for node in nodes:
            self.nodes.append(node)
        self.params = nodes[0].params
        self.BCs = BCs

    def infix(self):
        res = " | ".join([node.infix() for node in self.nodes])
        if self.BCs is not None:
            res = res + " -- "
            for k, v in self.BCs.items():
                res += f"{k}: {v}, "
            res = res[:-2]
        return res

    def __len__(self):
        return sum([len(node) for node in self.nodes])

    def prefix(self):
        res = ",|,".join([node.prefix() for node in self.nodes])
        if self.BCs is not None:
            res = res + ",--"
            for k, v in self.BCs.items():
                res += f",{k},{v}"
        return res

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return str(self)

    def val(self, u, dim, deterministic=True):
        batch_vals = [
            np.expand_dims(
                node.val(np.copy(u), deterministic=deterministic, space_dim=dim), -1
            )
            for node in self.nodes
        ]
        return np.concatenate(batch_vals, -1)

    def replace_node_value(self, old_value, new_value):
        for node in self.nodes:
            node.replace_node_value(old_value, new_value)
