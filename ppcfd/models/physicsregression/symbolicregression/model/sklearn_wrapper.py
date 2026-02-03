import copy
import math
import signal
import time
import traceback
from collections import defaultdict
from copy import deepcopy

import numpy as np
import paddle
from sklearn import feature_selection
from sklearn.base import BaseEstimator

from ..envs.generators import Node
from ..metrics import compute_metrics
from . import utils_wrapper


def corr(X, y, epsilon=1e-10):
    """
    X : shape n*d
    y : shape n
    """
    cov = y @ X / len(y) - y.mean() * X.mean(axis=0)
    corr = cov / (epsilon + X.std(axis=0) * y.std())
    return corr


def get_top_k_features(X, y, k=10):
    if y.ndim == 2:
        y = y[:, 0]
    if X.shape[1] <= k:
        return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        top_features = np.argsort(-np.abs(scores))
        print("keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])


def exchange_node_values(tree, dico):
    new_tree = copy.deepcopy(tree)
    for old, new in dico.items():
        new_tree.replace_node_value(old, new)
    return new_tree


class SymbolicTransformerRegressor(BaseEstimator):
    def __init__(
        self,
        model=None,
        max_input_points=10000,
        max_number_bags=-1,
        stop_refinement_after=1,
        n_trees_to_refine=1,
        rescale=True,
    ):
        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.model = model
        self.stop_refinement_after = stop_refinement_after
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def fit(self, X, Y, hints, verbose=False, refinement_types=["id"]):
        self.start_fit = time.time()
        if not isinstance(X, list):
            X = [X]
            Y = [Y]
            hints = [[h] for h in hints]
        n_datasets = len(X)
        self.top_k_features = [None for _ in range(n_datasets)]
        for i in range(n_datasets):
            self.top_k_features[i] = get_top_k_features(
                X[i], Y[i], k=self.model.env.params.max_input_dimension
            )
            X[i] = X[i][:, self.top_k_features[i]]
        scaler = utils_wrapper.StandardScaler() if self.rescale else None
        scale_params = {}
        if scaler is not None:
            scaled_X = []
            for i, x in enumerate(X):
                scaled_X.append(scaler.fit_transform(x))
                scale_params[i] = scaler.get_params()
        else:
            scaled_X = X
        inputs, inputs_ids, new_hints = [], [], [[] for i in range(len(hints))]
        for seq_id in range(len(scaled_X)):
            for seq_l in range(len(scaled_X[seq_id])):
                y_seq = Y[seq_id]
                if len(y_seq.shape) == 1:
                    y_seq = np.expand_dims(y_seq, -1)
                if seq_l % self.max_input_points == 0:
                    inputs.append([])
                    inputs_ids.append(seq_id)
                    for i, h in enumerate(new_hints):
                        h.append(copy.deepcopy(hints[i][seq_id]))
                inputs[-1].append([scaled_X[seq_id][seq_l], y_seq[seq_l]])
        if self.max_number_bags > 0:
            inputs = inputs[: self.max_number_bags]
            inputs_ids = inputs_ids[: self.max_number_bags]
            new_hints = [h[: self.max_number_bags] for h in new_hints]
        forward_time = time.time()
        outputs = self.model(inputs, new_hints)
        if verbose:
            print("Finished forward in {} secs".format(time.time() - forward_time))
        outputs, _, _ = outputs
        candidates = defaultdict(list)
        assert len(inputs) == len(outputs), "Problem with inputs and outputs"
        for i in range(len(inputs)):
            input_id = inputs_ids[i]
            candidate = outputs[i]
            candidates[input_id].extend(candidate)
        assert (
            len(candidates.keys()) == n_datasets
        ), "issue with n_can={}, n_can_ppl={}, n_data={}".format(
            len(candidates.keys()), n_datasets
        )
        self.tree = {}
        for input_id, candidates_id in candidates.items():
            if len(candidates_id) == 0:
                self.tree[input_id] = None
                continue
            if scaler is not None:
                rescaled_candidates = [
                    scaler.rescale_function(self.model.env, c, *scale_params[input_id])
                    for c in candidates_id
                ]
            else:
                rescaled_candidates = candidates_id
            refined_candidates = self.refine(
                X[input_id], Y[input_id], rescaled_candidates, verbose, refinement_types
            )
            self.tree[input_id] = refined_candidates

    @paddle.no_grad()
    def evaluate_tree(self, tree, X, y, metric):
        try:
            use_abs = tree.params.use_abs
            tree.params.use_abs = True
            y_tilde = tree.val(X).reshape((-1, 1))
            tree.params.use_abs = use_abs
        except:
            y_tilde = np.array([np.nan]).repeat(X.shape[0])
        metrics = compute_metrics(
            {"true": [y], "predicted": [y_tilde], "predicted_tree": [tree]},
            metrics=metric,
        )
        return metrics[metric][0]

    def order_candidates(self, X, y, candidates, metric="_mse", verbose=False):
        scores = []
        for candidate in candidates:
            if metric not in candidate:
                score = self.evaluate_tree(candidate["predicted_tree"], X, y, metric)
                if math.isnan(score):
                    score = np.inf if metric.startswith("_") else -np.inf
            else:
                score = candidates[metric]
            scores.append(score)
        ordered_idx = np.argsort(scores)
        if not metric.startswith("_"):
            ordered_idx = list(reversed(ordered_idx))
        candidates = [candidates[i] for i in ordered_idx]
        return candidates

    def refine(self, X, y, candidates, verbose, refinement_types):
        env = self.model.env
        generator = self.model.env.generator
        refined_candidates = []
        for i, candidate in enumerate(candidates):
            candidate_skeleton, candidate_constants = generator.function_to_skeleton(
                candidate, constants_with_idx=True
            )
            if "CONSTANT" in candidate_constants:
                candidates[i] = generator.wrap_equation_floats(
                    candidate_skeleton, np.random.randn(len(candidate_constants))
                )
        candidates = [
            {
                "refinement_type": "NoRef",
                "predicted_tree": candidate,
                "time": time.time() - self.start_fit,
            }
            for candidate in candidates
        ]
        candidates = self.order_candidates(
            X, y, candidates, metric="_mse", verbose=verbose
        )
        skeleton_candidates, candidates_to_remove = {}, []
        for i, candidate in enumerate(candidates):
            skeleton_candidate, _ = generator.function_to_skeleton(
                candidate["predicted_tree"], constants_with_idx=False
            )
            if skeleton_candidate.infix() in skeleton_candidates:
                candidates_to_remove.append(i)
            else:
                skeleton_candidates[skeleton_candidate.infix()] = 1
        if verbose:
            print(
                "Removed {}/{} skeleton duplicata".format(
                    len(candidates_to_remove), len(candidates)
                )
            )
        candidates = [
            candidates[i]
            for i in range(len(candidates))
            if i not in candidates_to_remove
        ]
        if self.n_trees_to_refine > 0:
            candidates_to_refine = candidates[: self.n_trees_to_refine]
        else:
            candidates_to_refine = copy.deepcopy(candidates)
        for candidate in candidates_to_refine:
            refinement_strategy = utils_wrapper.BFGSRefinement()
            tree_to_refine = candidate["predicted_tree"]
            refined_candidate = self._safely_refine(
                X, y, tree_to_refine, refinement_types
            )
            if refined_candidate is not None:
                refined_candidates.append(
                    {"refinement_type": "BFGS", "predicted_tree": refined_candidate}
                )
        candidates.extend(refined_candidates)
        candidates = self.order_candidates(X, y, candidates, metric="r2")
        for candidate in candidates:
            if "time" not in candidate:
                candidate["time"] = time.time() - self.start_fit
        return candidates

    def _safely_refine(self, X, y, node_to_refine, safety_types=["id"]):
        refined_nodes = []
        for safely_type in safety_types:
            if safely_type == "id":
                safely_y = y
                root = node_to_refine
                refined_node = self._refine(X, safely_y, root)
                refined_nodes.append(refined_node)
            elif safely_type == "neg":
                safely_y = deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                root = Node("neg", self.model.env.params, children=[safely_node])
                root.unit = safely_node.unit
                refined_node = self._refine(X, safely_y, root)
                refined_nodes.append(refined_node)
            elif safely_type == "inv":
                safely_y = 1 / deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                root = Node("inv", self.model.env.params, children=[safely_node])
                root.unit = -safely_node.unit
                refined_node = self._refine(X, safely_y, root)
                try:
                    refined_nodes.append(refined_node.children[0])
                except:
                    refined_nodes.append(safely_node)
            elif safely_type == "linear":
                safely_y = deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                child1 = Node("0", self.model.env.params)
                child2 = Node("1", self.model.env.params)
                child3 = Node(
                    "mul", self.model.env.params, children=[child2, safely_node]
                )
                root = Node("add", self.model.env.params, children=[child1, child3])
                child2.unit = np.zeros(5)
                child3.unit = safely_node.unit
                child1.unit = safely_node.unit
                root.unit = safely_node.unit
                refined_node = self._refine(X, safely_y, root)
                refined_nodes.append(refined_node)
            elif safely_type == "safe":
                safely_y = y
                root = deepcopy(node_to_refine)
                self.model.env.generator.unify_const(root)
                self.model.env.generator.unify_const3(root)
                refined_node = self._refine(X, safely_y, root, ignore=[0, 1, -1])
                refined_nodes.append(refined_node)
            elif safely_type == "safe-neg":
                safely_y = deepcopy(y)
                safely_node = deepcopy(node_to_refine)
                root = Node("neg", self.model.env.params, children=[safely_node])
                root.unit = safely_node.unit
                self.model.env.generator.unify_const(root)
                self.model.env.generator.unify_const3(root)
                refined_node = self._refine(X, safely_y, root, ignore=[0, 1, -1])
                refined_nodes.append(refined_node)
            else:
                raise ValueError(f"{safely_type} is not permitted")
        refined_nodes = self.order_tree(X, y, refined_nodes)
        return refined_nodes[0]

    def _refine(self, X, y, node_to_refine, ignore=[]):
        generator = self.model.env.generator
        refinement_strategy = utils_wrapper.BFGSRefinement()
        try:
            node_skeleton, node_constants = generator.function_to_skeleton(
                node_to_refine, constants_with_idx=True, ignore=ignore
            )
            refined_node, refine_success = refinement_strategy.go(
                env=self.model.env,
                tree=node_skeleton,
                coeffs0=node_constants,
                X=X,
                y=y,
                downsample=1024,
                stop_after=1,
            )
            assert refined_node is not None
        except:
            message = traceback.format_exc()
            refined_node = node_to_refine
        return refined_node

    def order_tree(self, X, y, trees, metric="_mse"):
        scores = []
        for tree in trees:
            score = self.evaluate_tree(tree, X, y, metric)
            if math.isnan(score):
                score = np.inf if metric.startswith("_") else -np.inf
            scores.append(score)
        ordered_idx = np.argsort(scores)
        if not metric.startswith("_"):
            ordered_idx = list(reversed(ordered_idx))
        trees = [trees[i] for i in ordered_idx]
        return trees

    def __str__(self):
        if hasattr(self, "tree"):
            for tree_idx in range(len(self.tree)):
                for gen in self.tree[tree_idx]:
                    print(gen)
        return "Transformer"

    def retrieve_refinements_types(self):
        return ["BFGS", "NoRef"]

    def exchange_tree_features(self):
        top_k_features = self.top_k_features
        for dataset_id, candidates in self.tree.items():
            exchanges = {}
            for i, feature in enumerate(top_k_features[dataset_id]):
                exchanges["x_{}".format(i)] = "x_{}".format(feature)
            if candidates is None:
                continue
            for candidate in candidates:
                candidate["relabed_predicted_tree"] = exchange_node_values(
                    candidate["predicted_tree"], exchanges
                )

    def retrieve_tree(
        self, refinement_type=None, dataset_idx=0, all_trees=False, with_infos=False
    ):
        self.exchange_tree_features()
        if dataset_idx == -1:
            idxs = [_ for _ in range(len(self.tree))]
        else:
            idxs = [dataset_idx]
        best_trees = []
        for idx in idxs:
            best_tree = copy.deepcopy(self.tree[idx])
            if best_tree and refinement_type is not None:
                best_tree = list(
                    filter(
                        lambda gen: gen["refinement_type"] == refinement_type, best_tree
                    )
                )
            if not best_tree:
                if with_infos:
                    best_trees.append(
                        {
                            "predicted_tree": None,
                            "refinement_type": refinement_type,
                            "time": None,
                            "perplexity": None,
                            "relabed_predicted_tree": None,
                        }
                    )
                else:
                    best_trees.append(None)
            elif with_infos:
                if all_trees:
                    best_trees.append(best_tree)
                else:
                    best_trees.append(best_tree[0])
            elif all_trees:
                best_trees.append(
                    [best_tree[i]["predicted_tree"] for i in range(len(best_tree))]
                )
            else:
                best_trees.append(best_tree[0]["predicted_tree"])
        if dataset_idx != -1:
            return best_trees[0]
        else:
            return best_trees

    def predict(
        self,
        X,
        refinement_type=None,
        tree_idx=0,
        batch=False,
        oracle_results=False,
        oracle_tree=None,
        merged_type=None,
        oracle_type=None,
    ):
        if not isinstance(X, list):
            X = [X]
        if oracle_results:
            for i in range(len(X)):
                X[i] = X[i][:, self.top_k_features[i]]
        else:
            for i in range(len(X)):
                X[i] = X[i][:, self.top_k_features[i]]
        res = []
        if batch:
            tree = self.retrieve_tree(refinement_type=refinement_type, dataset_idx=-1)
            for tree_idx in range(len(tree)):
                X_idx = X[tree_idx]
                if tree[tree_idx] is None:
                    res.append(None)
                else:
                    try:
                        node = tree[tree_idx]
                        use_abs = node.params.use_abs
                        node.params.use_abs = True
                        y = node.val(X_idx).reshape((-1, 1))
                        node.params.use_abs = use_abs
                    except:
                        y = np.array([np.nan]).repeat(X_idx.shape[0])
                    res.append(y)
            return res
        elif oracle_results:
            tree = oracle_tree
            for tree_idx in range(len(tree)):
                X_idx = X[tree_idx]
                if tree[tree_idx] is None:
                    res.append(None)
                else:
                    try:
                        node = tree[tree_idx]
                        use_abs = node.params.use_abs
                        node.params.use_abs = True
                        y = node.val(X_idx).reshape((-1, 1))
                        node.params.use_abs = use_abs
                    except:
                        y = np.array([np.nan]).repeat(X_idx.shape[0])
                    res.append(y)
            return res
        else:
            X_idx = X[tree_idx]
            tree = self.retrieve_tree(
                refinement_type=refinement_type, dataset_idx=tree_idx
            )
            if tree is not None:
                try:
                    use_abs = tree.params.use_abs
                    tree.params.use_abs = True
                    y = tree.val(X_idx).reshape((-1, 1))
                    tree.params.use_abs = use_abs
                except:
                    y = np.array([np.nan]).repeat(X_idx.shape[0])
                return y
            else:
                return None
