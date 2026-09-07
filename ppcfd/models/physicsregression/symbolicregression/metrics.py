from collections import defaultdict

import numpy as np
import scipy
import sympy as sp
import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from timeout_decorator import timeout


def compute_metrics(infos, metrics="r2"):
    results = defaultdict(list)
    if metrics == "":
        return {}
    if "true" in infos:
        true, predicted = infos["true"], infos["predicted"]
        assert len(true) == len(
            predicted
        ), "issue with len, true: {}, predicted: {}".format(len(true), len(predicted))
        for i in range(len(true)):
            if predicted[i] is None:
                continue
            if len(true[i].shape) == 2:
                true[i] = true[i][:, 0]
            if len(predicted[i].shape) == 2:
                predicted[i] = predicted[i][:, 0]
            assert (
                true[i].shape == predicted[i].shape
            ), "Problem with shapes: {}, {}".format(true[i].shape, predicted[i].shape)
    for metric in metrics.split(","):
        if metric == "r2":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        results[metric].append(r2_score(true[i], predicted[i]))
                    except Exception as e:
                        results[metric].append(np.nan)
        if metric == "r2_zero":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        results[metric].append(max(0, r2_score(true[i], predicted[i])))
                    except Exception as e:
                        results[metric].append(np.nan)
        elif metric.startswith("accuracy_l1"):
            if metric == "accuracy_l1":
                atol, rtol = 0.0, 0.1
                tolerance_point = 0.95
            elif metric == "accuracy_l1_biggio":
                atol, rtol = 0.001, 0.05
                tolerance_point = 0.95
            else:
                atol = 0
                rtol = float(metric.split("_")[-1])
                tolerance_point = 0.95
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        is_close = np.isclose(
                            predicted[i], true[i], atol=atol, rtol=rtol
                        )
                        results[metric].append(
                            float(is_close.mean() >= tolerance_point)
                        )
                    except Exception as e:
                        results[metric].append(np.nan)
        elif metric == "_mse":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        results[metric].append(
                            mean_squared_error(true[i], predicted[i])
                        )
                    except Exception as e:
                        results[metric].append(np.nan)
        elif metric == "_nmse":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        mean_y = np.mean(true[i])
                        NMSE = np.mean(np.square(true[i] - predicted[i])) / mean_y
                        results[metric].append(NMSE)
                    except Exception as e:
                        results[metric].append(np.nan)
        elif metric == "_rmse":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        results[metric].append(
                            mean_squared_error(true[i], predicted[i], squared=False)
                        )
                    except Exception as e:
                        results[metric].append(np.nan)
        elif metric == "_complexity":
            if "predicted_tree" not in infos:
                results[metric].extend([np.nan for _ in range(len(infos["true"]))])
                continue
            predicted_tree = infos["predicted_tree"]
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(len(predicted_tree[i].prefix().split(",")))
        elif metric == "_relative_complexity":
            if "tree" not in infos or "predicted_tree" not in infos:
                results[metric].extend([np.nan for _ in range(len(infos["true"]))])
                continue
            tree = infos["tree"]
            predicted_tree = infos["predicted_tree"]
            for i in range(len(predicted_tree)):
                if predicted_tree[i] is None:
                    results[metric].append(np.nan)
                else:
                    results[metric].append(
                        len(predicted_tree[i].prefix().split(","))
                        - len(tree[i].prefix().split(","))
                    )
        elif metric == "is_symbolic_solution":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        diff = true[i] - predicted[i]
                        div = true[i] / (predicted[i] + 1e-100)
                        std_diff = scipy.linalg.norm(np.abs(diff - diff.mean(0)))
                        std_div = scipy.linalg.norm(np.abs(div - div.mean(0)))
                        if std_diff < 1e-10 and std_div < 1e-10:
                            results[metric].append(1.0)
                        else:
                            results[metric].append(0.0)
                    except Exception as e:
                        results[metric].append(np.nan)
        elif metric == "_l1_error":
            true, predicted = infos["true"], infos["predicted"]
            for i in range(len(true)):
                if predicted[i] is None or np.isnan(np.min(predicted[i])):
                    results[metric].append(np.nan)
                else:
                    try:
                        l1_error = np.mean(np.abs(true[i] - predicted[i]))
                        if np.isnan(l1_error):
                            results[metric].append(np.inf)
                        else:
                            results[metric].append(l1_error)
                    except Exception as e:
                        results[metric].append(np.nan)
    return results


@timeout(10)
def parse(expr, local_dic=None):
    return sp.parse_expr(expr, local_dict=local_dic)


@timeout(10)
def simplify(expr):
    return sp.simplify(expr)


@timeout(10)
def value_approximate(expr, eps=6):
    if expr == sp.nan:
        return sp.parse_expr("0")
    try:
        v = expr.xreplace(
            {
                n: (round(n, eps) if abs(n - round(n)) > 10 ** -eps else round(n))
                for n in expr.atoms(sp.Number)
            }
        )
    except:
        return sp.parse_expr("0")
    if v == sp.nan:
        return sp.parse_expr("0")
    return v


@timeout(10)
def reduce_abs(expr):
    return expr.replace(sp.Abs, lambda x: x)


@timeout(10)
def expr_to_prefix(expr):
    if isinstance(expr, str):
        expr = parse(expr)
    if expr.is_Atom:
        return str(expr)
    operator_map = {sp.Add: "+", sp.Mul: "*", sp.Pow: "**"}
    operator = operator_map.get(expr.func, str(expr.func))
    if operator == "**" and expr.args[0].func == sp.Pow:
        operands = expr_to_prefix(expr.args[0].args[0])
        coefficient = expr.args[1] * expr.args[0].args[1]
        return f"**,{operands},{coefficient}"
    elif operator == "**" and expr.args[-1] == 0.5 and expr.args[0].func == sp.exp:
        operands = expr_to_prefix(expr.args[0].args[0])
        return f"exp,*,0.5,{operands}"
    elif (
        operator == "**"
        and expr.args[-1] == 0.5
        and expr.args[0].func == sp.Abs
        and expr.args[0].args[0].func == sp.exp
    ):
        operands = expr_to_prefix(expr.args[0].args[0].args[0])
        return f"exp,*,0.5,{operands}"
    elif operator == "**" and expr.args[-1] == 0.5 and expr.args[0].func == sp.Mul:
        operands = [expr_to_prefix(sp.sqrt(arg)) for arg in expr.args[0].args]
        return f"{','.join(['*'] * max(1, len(operands) - 1))},{','.join(operands)}"
    elif operator == "Abs" and expr.args[0].func == sp.Pow:
        return expr_to_prefix(expr.args[0])
    elif expr.func == sp.re:
        return expr_to_prefix(expr.args[0])
    elif operator == "**" and expr.args[-1] == 0.5:
        operands = [expr_to_prefix(sp.Abs(expr.args[0])), "0.5"]
        return f"{operator},{','.join(operands)}"
    operands = [expr_to_prefix(arg) for arg in expr.args]
    return f"{','.join([operator] * max(1, len(operands) - 1))},{','.join(operands)}"


@timeout(10)
def prefix_to_infix(prefix):
    l = prefix.split(",")

    def _dfs(tokens):
        token = tokens.pop(0)
        if token in ("+", "*", "**", "-", "/"):
            left = _dfs(tokens)
            right = _dfs(tokens)
            return f"(({left}) {token} ({right}))"
        elif token in (
            "sin",
            "cos",
            "sqrt",
            "tan",
            "log",
            "arcsin",
            "arccos",
            "arctan",
            "inv",
            "neg",
            "exp",
            "Abs",
            "abs",
            "sinh",
            "cosh",
            "tanh",
        ):
            operand = _dfs(tokens)
            return f"{token}({operand})"
        else:
            return token

    return _dfs(l)


def cal_sym_acc(pred_exprs, true_exprs, use_tqdm_bar=False):
    for i in range(len(pred_exprs)):
        for k, v in {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "inv": "1/",
            "neg": "-",
            "pow": "**",
            "pi": str(round(np.pi, 10)),
        }.items():
            pred_exprs[i] = str(pred_exprs[i]).replace(k, v)
            true_exprs[i] = str(true_exprs[i]).replace(k, v)
        pred_exprs[i] = pred_exprs[i].replace("nan", "0")
        pred_exprs[i] = pred_exprs[i].replace("NaN", "0")
        try:
            pred = value_approximate(parse(pred_exprs[i]), eps=5)
            prefix = expr_to_prefix(pred)
            infix = prefix_to_infix(prefix)
            pred = parse(infix)
            pred = value_approximate(pred, eps=5)
            prefix = expr_to_prefix(pred)
            infix = prefix_to_infix(prefix)
            pred_exprs[i] = str(value_approximate(parse(infix), eps=5))
        except:
            pred_exprs[i] = "0"
        true = value_approximate(parse(true_exprs[i]), eps=5)
        prefix = expr_to_prefix(true)
        infix = prefix_to_infix(prefix)
        true_exprs[i] = str(value_approximate(parse(infix), eps=5))
    if use_tqdm_bar:
        pbar = tqdm.tqdm(total=len(pred_exprs))
    sym_acc = []
    for i in range(len(pred_exprs)):
        if pred_exprs[i] == "0":
            sym_acc.append(0)
            if use_tqdm_bar:
                pbar.update(1)
            continue
        try:
            p = parse(pred_exprs[i])
            t = parse(true_exprs[i])
            d1 = simplify(p - t).evalf()
            d2 = simplify(p / t).evalf()
            d1_s = value_approximate(d1, 3)
            d2_s = value_approximate(d2, 3)
            d1_s = parse(str(d1_s))
            d2_s = parse(str(d2_s))
            d1_ss = simplify(d1_s)
            d2_ss = simplify(d2_s)
            _sym_acc = int(d1_ss.is_number or d2_ss.is_number)
            if d2_ss == 0:
                _sym_acc = 0
            elif _sym_acc == 0:
                d1_sss = expr_to_prefix(d1_ss)
                d1_sss = parse(prefix_to_infix(d1_sss))
                d1_sss = parse(str(value_approximate(d1_sss, eps=3)))
                d1_ssss = simplify(d1_sss)
                d2_sss = expr_to_prefix(d2_ss)
                d2_sss = parse(prefix_to_infix(d2_sss))
                d2_sss = parse(str(value_approximate(d2_sss, eps=3)))
                d2_ssss = simplify(d2_sss)
                d1_sssss = simplify(reduce_abs(d1_ssss))
                d2_sssss = simplify(reduce_abs(d2_ssss))
                if d2_sssss == 0:
                    _sym_acc = 0
                else:
                    _sym_acc = int(d1_sssss.is_number or d2_sssss.is_number)
        except:
            import traceback

            message = traceback.format_exc()
            if "Timed Out" in message:
                pass
            else:
                pass
            _sym_acc = 0
        sym_acc.append(_sym_acc)
        if use_tqdm_bar:
            pbar.update(1)
    if use_tqdm_bar:
        pbar.close()
    return sym_acc
