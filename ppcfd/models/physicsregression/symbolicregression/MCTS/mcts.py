import time
from copy import deepcopy

import numpy as np
import sympy as sp
import tqdm
from scipy.optimize import minimize

np.seterr(all="ignore")
NON_TERMINAL = ["O", "A", "B"]
TERMINAL = ["C"]


def production_to_expr(production):
    s = "O"
    for source, target in production:
        s = s.replace(source, target, 1)
    return s


class MctsNode:
    def __init__(self, productions, nt, grammar, non_terminal, parent=None):
        self.productions = productions
        self.expr = production_to_expr(productions)
        self.nt = nt
        self.parent = parent
        self.grammar = grammar
        self.non_terminal = non_terminal
        self.visits = 0
        self.rewards = 0
        self.is_terminal = nt == []
        self.is_expandable = not self.is_terminal
        self.children = (
            [(0) for _ in range(len(grammar[nt[0]]))] if not self.is_terminal else []
        )
        self.num_children = 0

    def push_children(self, idx, children):
        self.children[idx] = children
        self.num_children += 1
        if self.num_children == len(self.grammar[self.nt[0]]):
            self.is_expandable = False

    def __repr__(self):
        s = "Node\nexpr:{}\nchildren:{}\nvisits:{}\nrewards:{}".format(
            self.expr, len(self.children), self.visits, self.rewards
        )
        return s


class MCTS:
    def __init__(
        self,
        params,
        env,
        base_grammar,
        non_terminal,
        t_max=100,
        expr_max=60,
        discount_factor=0.999,
        mcts_print_freq=100,
        uct_const=1 / np.sqrt(2),
        search_round=100000,
        num_simulations=10,
        greedy_eps=0.1,
        early_stop=1,
        early_stop_after=100,
        mcts_search_type="normal",
    ):
        self.params = params
        self.env = env
        self.grammar = base_grammar
        self.non_terminal = non_terminal
        self.mcts_search_type = mcts_search_type
        self.t_max = t_max
        self.expr_max = expr_max
        self.discount_factor = discount_factor
        self.print_freq = mcts_print_freq
        self.UCT_const = uct_const
        self.search_round = search_round
        self.num_simulations = num_simulations
        self.greedy_eps = greedy_eps
        self.early_stop = early_stop
        self.early_stop_after = early_stop_after

    def unify_same_value(self, nodes):
        dfs_stack = deepcopy(nodes)
        res = deepcopy(nodes)
        while dfs_stack:
            current_node = dfs_stack.pop(0)
            if current_node is None:
                continue
            for n in current_node.children:
                if n.value == nodes[0].value:
                    res.append(n)
                    dfs_stack.append(n)
        return res

    def dfs_value_chain(self, node, chain, fill_value="A"):
        """given value chain, dfs search target node and fill the leaf with fill_value"""

        def _dfs(node, chain):
            if node.value != chain[0]:
                chain = chain[1:]
            if len(chain) > 0:
                for n in node.children:
                    _dfs(n, chain)
            else:
                if fill_value == "auto":
                    if len(node) < 2:
                        node.value = "B"
                    else:
                        node.value = "A"
                else:
                    node.value = fill_value
                node.children = []

        _dfs(node, chain)

    def extract_subexpr(self, node, update=False):
        if node is None:
            return
        try:
            dfs_stack = [node]
            sub_expr1 = []
            while dfs_stack:
                current_node = dfs_stack.pop(0)
                if "x_" in str(current_node) and len(current_node) > 2:
                    sub_expr1.append(str(current_node))
                if current_node is not None:
                    dfs_stack.extend(current_node.children)
            dfs_stack = [node]
            sub_expr2 = []
            while dfs_stack:
                current_node = dfs_stack.pop(0)
                if current_node is None:
                    continue
                value_chain = []
                current_nodes = [current_node]
                history_nodes = []
                while True:
                    current_nodes = self.unify_same_value(current_nodes)
                    history_nodes.append(current_nodes)
                    value_chain.append(current_nodes[0].value)
                    s = set([cc.value for c in current_nodes for cc in c.children])
                    if len(s) > 2 or any(
                        [(len(cc) < 2) for c in current_nodes for cc in c.children]
                        + [(len(c) < 2) for c in current_nodes]
                    ):
                        break
                    current_nodes = [
                        cc
                        for c in current_nodes
                        for cc in c.children
                        if cc.value != value_chain[-1]
                    ]
                for _num in range(3, len(value_chain) + 1):
                    target_chain = value_chain[:_num]
                    target_node = deepcopy(current_node)
                    self.dfs_value_chain(target_node, target_chain, fill_value="auto")
                    sub_expr2.append(str(target_node))
                dfs_stack.extend(current_node.children)
            for i in range(len(sub_expr1)):
                for k, v in {
                    "add": "+",
                    "sub": "-",
                    "mul": "*",
                    "inv": "1/",
                    "neg": "-",
                }.items():
                    sub_expr1[i] = sub_expr1[i].replace(k, v)
            for i in range(len(sub_expr2)):
                for k, v in {
                    "add": "+",
                    "sub": "-",
                    "mul": "*",
                    "inv": "1/",
                    "neg": "-",
                }.items():
                    sub_expr2[i] = sub_expr2[i].replace(k, v)
            if update:
                self.grammar["A"].extend(sub_expr2)
                self.grammar["B"].extend(sub_expr1)
            return sub_expr1, sub_expr2
        except:
            pass

    def _is_float(self, f):
        try:
            _ = float(f)
            return True
        except:
            return False

    def extract_target_str(self, tree):
        if tree is None:
            return ["A"], [1]
        dfs_stack = [tree]
        target_str = []
        target_length = []
        while dfs_stack:
            current_node = dfs_stack.pop(0)
            if hasattr(current_node, "extract"):
                del current_node.extract
            else:
                _value = current_node.value
                _children = current_node.children
                current_node.value = "A"
                current_node.children = []
                _str = str(tree)
                _length = len(tree)
                current_node.value = _value
                current_node.children = _children
                for k, v in {
                    "add": "+",
                    "sub": "-",
                    "mul": "*",
                    "inv": "1/",
                    "neg": "-",
                }.items():
                    _str = _str.replace(k, v)
                target_str.append(_str)
                target_length.append(_length)
            dfs_stack.extend(current_node.children)
            if _value in ["add", "sub", "mul"] and self._is_float(_children[0].value):
                _children[1].extract = False
        return target_str, target_length

    def extract_original_solution(self, tree, xy):
        if tree is not None:
            e = str(tree)
            for k, v in {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "inv": "1/",
                "neg": "-",
                "pow": "**",
            }.items():
                e = e.replace(k, v)
            solutions = [["", e, self.reward(e, xy, 1)]]
        else:
            solutions = []
        return solutions

    def const_optimize(self, expr, xy, n_restart=5):
        """
        we just optimize the 'skeleton' operator in the expression
        """
        x, y = xy
        dimension = expr.count("C")
        if dimension == 0:
            return expr, 1
        expr = expr.replace("^", "**")
        for i in range(dimension):
            expr = expr.replace("C", "c_{}".format(i), 1)
        x_variables = [sp.Symbol("x_{}".format(i)) for i in range(x.shape[1])]
        c_variables = [sp.Symbol("c_{}".format(i)) for i in range(dimension)]
        func = sp.lambdify(x_variables + c_variables, expr, "numpy")

        def loss(c, x, y):
            pred = np.array([func(*x.T, *c)])
            return np.linalg.norm(pred.reshape(-1) - y.reshape(-1), 2)

        F_loss = []
        consts = []
        for i in range(n_restart):
            try:
                x0 = np.random.randn(dimension)
                res = minimize(loss, x0=x0, args=(x, y), method="bfgs", tol=1e-06)
                if res.success:
                    consts.append(res.x)
                    F_loss.append(loss(res.x, x, y))
            except:
                pass
        if consts != []:
            idx = np.argmin(F_loss)
            for i in range(dimension, 0, -1):
                expr = expr.replace(f"c_{i}", f"({consts[idx][i - 1]:.4f})")
            return expr, 1
        else:
            return expr, 0

    def UCT(self, parent, child):
        if self.mcts_search_type == "normal":
            reward = child.rewards / child.visits
        elif self.mcts_search_type == "max":
            reward = child.rewards
        parismony = self.UCT_const * np.sqrt(np.log(parent.visits) / child.visits)
        return reward + parismony

    def mse(self, expr, xy):
        x, y = xy
        num_variables = x.shape[1]
        update_expr = expr.replace("pow", "**")
        x_variables = [sp.Symbol("x_{}".format(i)) for i in range(num_variables)]
        func = sp.lambdify(x_variables, update_expr, "numpy")
        try:
            y_pred = func(*x.T)
            return np.linalg.norm(y.reshape(-1) - y_pred.reshape(-1), 2)
        except:
            return np.nan

    def reward(self, expr, xy, l, f=None):
        x, y = xy
        num_variables = x.shape[1]
        x_variables = [sp.Symbol("x_{}".format(i)) for i in range(num_variables)]
        update_expr = expr.replace("pow", "**")
        func = sp.lambdify(x_variables, update_expr, "numpy")
        if f is None:
            f = self.discount_factor
        try:
            y_pred = func(*x.T)
            norm = np.linalg.norm(y.reshape(-1) - y_pred.reshape(-1), 2)
            R = float(f**l / np.sqrt(1 + norm**2 / y.shape[0]))
            return R
        except:
            import traceback

            message = traceback.format_exc()
            return 0

    def reward(self, expr, xy, l, f=None):
        from numpy import (abs, arccos, arccosh, arcsin, arcsinh, arctan,
                           arctanh, cos, cosh, exp, log, pi, sin, sinh, tan,
                           tanh)

        sqrt = lambda a: np.sqrt(np.abs(a))
        x, y = xy
        num_variables = x.shape[1]
        for i in range(num_variables):
            globals()[f"x_{i}"] = x[:, i]
        try:
            update_expr = expr.replace("pow", "**")
            y_pred = np.array(eval(update_expr))
            if f is None:
                f = self.discount_factor
            R = float(
                f**l
                / np.sqrt(
                    1
                    + np.linalg.norm(y.reshape(-1) - y_pred.reshape(-1), 2) ** 2
                    / y.shape[0]
                )
            )
            R2 = float(
                f**l
                / np.sqrt(
                    1
                    + np.linalg.norm(
                        (y.reshape(-1) - y_pred.reshape(-1))
                        / np.maximum(np.abs(y.reshape(-1)), 1e-10),
                        2,
                    )
                    ** 2
                    / y.shape[0]
                )
            )
            return max(R, R2)
        except:
            import traceback

            message = traceback.format_exc()
            return 0

    def selection(self, root):
        node = root
        t = 0
        while not node.is_expandable and not node.is_terminal and t < self.t_max:
            UCTs = np.array(
                [
                    (self.UCT(node, child) if not child.is_terminal else 0)
                    for child in node.children
                ]
            )
            p = np.zeros(len(node.children)) + self.greedy_eps / len(node.children)
            p[np.argmax(UCTs)] += 1
            p[UCTs == 0] = 0
            p /= np.sum(p)
            idx = np.random.choice(range(len(node.children)), p=p)
            node = node.children[idx]
            t += 1
        if node.is_terminal or t >= self.t_max:
            return node, False
        else:
            return node, True

    def _next_single_nt(self, nt, op):
        new_nt = [i for i in op if i in self.non_terminal]
        nt.pop(0)
        nt = new_nt + nt
        return nt

    def _next_nt(self, nt, op):
        if isinstance(op, str):
            return self._next_single_nt(nt, op)
        elif isinstance(op, list):
            for p in op:
                nt = self._next_single_nt(nt, p[1])
            return nt

    def expansion(self, node, nt):
        s = nt[0]
        valid_rules = self.grammar[s]
        p = np.array(
            [(1 if child == 0 else 0) for child in node.children], dtype=np.float32
        )
        p /= np.sum(p)
        idx = np.random.choice(range(len(valid_rules)), p=p)
        op = valid_rules[idx]
        if isinstance(op, str):
            nt = self._next_nt(nt, op)
            new_node = MctsNode(
                node.productions + [[s, op]],
                deepcopy(nt),
                self.grammar,
                self.non_terminal,
                parent=node,
            )
        elif isinstance(op, list):
            nt = self._next_nt(nt, op)
            new_node = MctsNode(
                node.productions + op,
                deepcopy(nt),
                self.grammar,
                self.non_terminal,
                parent=node,
            )
        node.push_children(idx, new_node)
        return new_node, nt

    def _simulation(self, expr, l, nt):
        t = 0
        productions = []
        while t + l <= self.expr_max:
            if nt == []:
                break
            s = nt[0]
            idx = np.random.randint(len(self.grammar[s]))
            op = self.grammar[s][idx]
            if isinstance(op, str):
                expr = expr.replace(s, op, 1)
                productions.append([s, op])
                nt = self._next_nt(nt, op)
                t += 1
            elif isinstance(op, list):
                for p in op:
                    expr = expr.replace(p[0], p[1], 1)
                    productions.append([p[0], p[1]])
                nt = self._next_nt(nt, op)
                t += len(op)
            else:
                print(f"unknown op type: {type(op)}")
                print(op)
        if nt != []:
            return None, productions
        return expr, productions

    def simulation(self, node, xy, nt):
        expr = node.expr
        max_reward = 0
        best_expr = None
        best_productions = []
        if node.is_terminal:
            optimized_expr, flag = self.const_optimize(expr, xy, n_restart=1)
            if flag == 0:
                reward = 0
            else:
                reward = self.reward(optimized_expr, xy, l=len(node.productions))
            if reward > max_reward:
                max_reward = reward
                best_expr = optimized_expr
                best_productions = []
            return max_reward, best_expr, node.productions + best_productions
        for i in range(self.num_simulations):
            simulation_expr, productions = self._simulation(
                expr, len(node.productions), deepcopy(nt)
            )
            if simulation_expr is None:
                continue
            optimized_expr, flag = self.const_optimize(simulation_expr, xy, n_restart=1)
            if flag == 0:
                reward = 0
            else:
                reward = self.reward(
                    optimized_expr, xy, l=len(productions) + len(node.productions)
                )
            if reward > max_reward:
                max_reward = reward
                best_expr = optimized_expr
                best_productions = productions
        return max_reward, best_expr, node.productions + best_productions

    def backpropagation(self, node, rewards):
        while node:
            if self.mcts_search_type == "normal":
                node.rewards += rewards
            elif self.mcts_search_type == "max":
                node.rewards = max(node.rewards, rewards)
            node.visits += 1
            node = node.parent

    def search(
        self, xy, productions=[["O", "A"]], nt=["A"], verbose=False, max_time=-1
    ):
        start_time = time.time()
        self.root = MctsNode(
            productions=productions,
            nt=nt,
            grammar=self.grammar,
            non_terminal=self.non_terminal,
        )
        self.best_solution = [[], "nothing", 0]
        early_stop_cnt = 0
        cnt = [0, 0, 0]
        self.print_freq = 100
        for search_round in range(1, self.search_round + 1):
            if search_round % self.print_freq == 0 and self.print_freq and verbose:
                print(
                    f"\rEpisode {search_round}/{self.search_round}, current best reward {self.best_solution[2]:.4f}.",
                    end="",
                )
            node, flag = self.selection(self.root)
            nt = deepcopy(node.nt)
            if flag:
                cnt[0] += 1
                node, nt = self.expansion(node, nt)
                max_reward, best_expr, best_productions = self.simulation(node, xy, nt)
                if max_reward > self.best_solution[2]:
                    stack = [self.root]
                    while stack:
                        temp = stack.pop(0)
                        temp.rewards *= self.best_solution[2] / max_reward
                        stack += [child for child in temp.children if child != 0]
                    self.best_solution = [best_productions, best_expr, max_reward]
                self.backpropagation(node, max_reward)
            elif node.is_terminal:
                cnt[1] += 1
                expr, flag = self.const_optimize(node.expr, xy, n_restart=1)
                if flag == 0:
                    reward = 0
                else:
                    reward = self.reward(expr, xy, l=len(node.productions))
                self.backpropagation(node, reward)
            else:
                cnt[2] += 1
                self.backpropagation(node, 0)
            if self.best_solution[2] > self.early_stop:
                early_stop_cnt += 1
            if early_stop_cnt >= self.early_stop_after:
                break
            if max_time > 0 and time.time() - start_time > max_time:
                break
        return self.best_solution
