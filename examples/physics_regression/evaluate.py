import copy
import json
import os
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from ppcfd.models.physicsregression.Oracle.oracle import Oracle
from ppcfd.models.physicsregression.symbolicregression.GA.ga import GeneticProgramming
from ppcfd.models.physicsregression.symbolicregression.GA.ga import TreeGenerator
from ppcfd.models.physicsregression.symbolicregression.MCTS.mcts import MCTS
from ppcfd.models.physicsregression.symbolicregression.metrics import cal_sym_acc
from ppcfd.models.physicsregression.symbolicregression.metrics import compute_metrics
from ppcfd.models.physicsregression.symbolicregression.model.model_wrapper import ModelWrapper
from ppcfd.models.physicsregression.symbolicregression.model.sklearn_wrapper import \
    SymbolicTransformerRegressor
from tqdm import tqdm

np.seterr(all="ignore")


def _seq_to_numpy(seq):
    """Ensure a batch of per-expression arrays is numpy, not paddle.Tensor.

    The whole symbolic pipeline downstream (Oracle, GA, MCTS, sklearn_wrapper)
    is numpy-based, but x_to_fit/y_to_fit may arrive as paddle.Tensor in this
    ported codebase. Convert once at this boundary instead of patching every
    numpy call site.
    """
    out = []
    for v in seq:
        if hasattr(v, "detach"):  # paddle.Tensor
            v = v.detach().cpu().numpy()
        out.append(v)
    return out


symbolic_str_dic = {
    "add": "+",
    "mul": "*",
    "sub": "-",
    "pow": "**",
    "inv": "1/",
    "neg": "-",
}


def read_file(filename, label="target", sep=None):
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]
    return X, y, feature_names


class Evaluator(object):
    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def set_env_copies(self, data_types):
        exprs = {
            dataset_name: None
            for dataset_name in [
                "exprs_train",
                "exprs_valid",
                "exprs_test",
                "sub_exprs_train",
                "sub_exprs_valid",
            ]
        }
        for dataset_name in exprs.keys():
            exprs[dataset_name] = getattr(self.env.generator, dataset_name)
            setattr(self.env.generator, dataset_name, None)
        for data_type in data_types:
            new_env = deepcopy(self.env)
            for dataset_name, dataset in exprs.items():
                setattr(new_env.generator, dataset_name, dataset)
            setattr(self, "{}_env".format(data_type), new_env)
        for dataset_name, dataset in exprs.items():
            setattr(self.env.generator, dataset_name, dataset)

    def evaluate_e2e(
        self,
        data_type,
        task,
        verbose=False,
        save=False,
        logger=None,
        save_file=None,
        datatype="train",
        verbose_res=False,
        refinement_types=["id", "neg", "inv", "linear", "safe", "safe-neg"],
        epoch_id=None
    ):
        assert datatype in ["train", "valid", "test", "valid-sub"]
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": self.trainer.epoch, "datatype": datatype})
        params = self.params
        logger.info(f"====== STARTING EVALUATION E2E:{datatype.upper()} =======")
        embedder = self.modules["embedder"]
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        embedder.eval()
        encoder.eval()
        decoder.eval()
        env = getattr(self, "{}_env".format(data_type))
        eval_size_per_gpu = params.eval_size
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            input_length_modulo=params.eval_input_length_modulo,
            test_env_seed=self.params.test_env_seed,
            datatype=datatype,
        )
        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )
        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )
        first_write = True
        if save:
            if save_file is None:
                save_file = (
                    self.params.eval_dump_path
                    if self.params.eval_dump_path is not None
                    else self.params.dump_path
                )
                if not os.path.exists(save_file):
                    os.makedirs(save_file)
                save_file = os.path.join(save_file, f"eval_e2e_{datatype}_{epoch_id}.csv")
        pbar = tqdm(total=eval_size_per_gpu)
        batch_results = defaultdict(list)
        sampling_time = time.time()
        for samples, _ in iterator:
            if verbose:
                print(
                    "Finished sampling in {} secs".format(time.time() - sampling_time)
                )
            starting_time = time.time()
            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]
            infos = samples["infos"]
            tree = samples["tree"]
            real_variables = samples["real_variables"]
            hints = []
            if params.use_hints:
                for used_hints in params.use_hints.split(","):
                    if used_hints == "units":
                        hints.append(samples[used_hints])
                    elif used_hints == "complexity":
                        hints.append([[0] for _ in range(len(samples[used_hints]))])
                    else:
                        hints.append([[] for _ in range(len(samples[used_hints]))])
            dstr.fit(
                x_to_fit,
                y_to_fit,
                hints,
                verbose=verbose,
                refinement_types=refinement_types,
            )
            for k, v in infos.items():
                infos[k] = v.tolist()
            best_gens = copy.deepcopy(
                dstr.retrieve_tree(
                    refinement_type="BFGS", dataset_idx=-1, with_infos=True
                )
            )
            predicted_tree = [best_gen["predicted_tree"] for best_gen in best_gens]
            batch_results["predicted_tree"].extend(
                [str(_tree) for _tree in predicted_tree]
            )
            batch_results["tree"].extend([str(_tree) for _tree in tree])
            y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type="BFGS", batch=True)
            assert len(y_to_fit) == len(y_tilde_to_fit)
            results_fit = compute_metrics(
                {
                    "true": y_to_fit,
                    "predicted": y_tilde_to_fit,
                    "tree": tree,
                    "predicted_tree": predicted_tree,
                },
                metrics=params.validation_metrics,
            )
            for k, v in results_fit.items():
                batch_results[k + "_fit"].extend(v)
            del results_fit
            if self.params.prediction_sigmas is None:
                prediction_sigmas = []
            else:
                prediction_sigmas = [
                    float(sigma) for sigma in self.params.prediction_sigmas.split(",")
                ]
            for sigma in prediction_sigmas:
                x_to_predict = samples["x_to_predict_{}".format(sigma)]
                y_to_predict = samples["y_to_predict_{}".format(sigma)]
                y_tilde_to_predict = dstr.predict(
                    x_to_predict, refinement_type="BFGS", batch=True
                )
                results_predict = compute_metrics(
                    {
                        "true": y_to_predict,
                        "predicted": y_tilde_to_predict,
                        "tree": tree,
                        "predicted_tree": predicted_tree,
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_predict.items():
                    batch_results[k + "_predict_{}".format(sigma)].extend(v)
                del results_predict
            for t1, t2, v in zip(tree, predicted_tree, real_variables):
                transition_dic = {f"x_{i}": _v for i, _v in enumerate(v)}
                transition_dic.update(symbolic_str_dic)
                if t1 is not None:
                    node1 = t1.copy()
                    self.env.generator.apply_transition(node1, transition_dic)
                    infix = str(node1)
                    batch_results["tree_labeled"].append(infix)
                else:
                    batch_results["tree_labeled"].append(None)
                if t2 is not None:
                    node2 = t2.copy()
                    self.env.generator.apply_transition(node2, transition_dic)
                    infix = str(node2)
                    batch_results["predicted_tree_labeled"].append(infix)
                else:
                    batch_results["predicted_tree_labeled"].append(None)
            for _tree, _unit in zip(predicted_tree, samples["units"]):
                if _tree is not None:
                    units_pred = env.equation_encoder.check_units(_tree, _unit)
                else:
                    units_pred = False
                batch_results["units_pred"].append(units_pred)
            batch_results["elapsed_time"].extend(
                [(time.time() - starting_time) / len(x_to_fit)] * len(x_to_fit)
            )
            for k in params.use_hints.split(","):
                v = samples[k]
                batch_results["hints_" + k].extend(v)
            if save:
                batch_results = pd.DataFrame.from_dict(batch_results)
                if first_write:
                    batch_results.to_csv(save_file, index=False)
                else:
                    batch_results.to_csv(save_file, mode="a", header=False, index=False)
            else:
                batch_results = pd.DataFrame.from_dict(batch_results)
                if first_write:
                    total_results = batch_results
                else:
                    total_results = pd.concat(
                        [total_results, batch_results], axis=0, ignore_index=True
                    )
            first_write = False
            batch_results = defaultdict(list)
            bs = len(x_to_fit)
            pbar.update(bs)
        if save:
            try:
                df = pd.read_csv(save_file, na_filter=True)
            except:
                logger.info("WARNING: no results")
                return
        else:
            df = total_results
        df = df.fillna(0)
        scores = {}
        print_columns = [
            f"r2_{b}"
            for b in [
                "fit",
                "predict_1.0",
                "predict_2.0",
                "predict_4.0",
                "predict_8.0",
                "predict_16.0",
            ]
        ]
        for k in print_columns:
            v = (np.array(list(df[k])) > 0.99).mean()
            kk = f"{k}>0.99"
            scores[kk] = v
            v = (np.array(list(df[k])) > 0.999).mean()
            kk = f"{k}>0.999"
            scores[kk] = v
        print_columns = [
            f"{a}_{b}"
            for a in [
                "r2_zero",
                "r2",
                "accuracy_l1_biggio",
                "accuracy_l1_1e-3",
                "accuracy_l1_1e-2",
                "accuracy_l1_1e-1",
            ]
            for b in [
                "fit",
                "predict_1.0",
                "predict_2.0",
                "predict_4.0",
                "predict_8.0",
                "predict_16.0",
            ]
        ] + ["_complexity_fit", "units_pred", "elapsed_time"]
        for k in print_columns:
            v = df[k].mean()
            scores[k] = v
        if verbose_res:
            for k, v in scores.items():
                print(f"{k:35s}: {v:.4f}")
        return scores

    def evaluate_oracle_mcts(
        self,
        data_type,
        task,
        verbose=False,
        logger=None,
        save_file=None,
        datatype="train",
        oracle_name="training",
        verbose_res=False,
    ):
        assert datatype in ["train", "valid", "test", "valid-sub"]
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": self.trainer.epoch, "datatype": datatype})
        params = self.params
        logger.info(f"====== STARTING EVALUATION MCTS:{datatype.upper()} =======")
        embedder = self.modules["embedder"]
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        embedder.eval()
        encoder.eval()
        decoder.eval()
        env = getattr(self, "{}_env".format(data_type))
        eval_size_per_gpu = params.eval_size
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            input_length_modulo=params.eval_input_length_modulo,
            test_env_seed=self.params.test_env_seed,
            datatype=datatype,
        )
        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )
        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )
        first_write = True
        current_pos = 0
        pbar = tqdm(total=eval_size_per_gpu)
        batch_results = defaultdict(list)
        oracle = Oracle(env, env.generator, params)
        for samples, _ in iterator:
            if current_pos < params.current_eval_pos:
                current_pos += len(samples["x_to_fit"])
                continue
            time1 = time.time()
            x_to_fit = _seq_to_numpy(samples["x_to_fit"])
            y_to_fit = _seq_to_numpy(samples["y_to_fit"])
            infos = samples["infos"]
            tree = samples["tree"]
            real_variables = samples["real_variables"]
            expr_idxs = infos["expr_idx"]
            hints = []
            for used_hints in params.use_hints.split(","):
                if used_hints == "units":
                    hints.append(samples[used_hints])
                elif used_hints == "complexity":
                    hints.append([[0] for _ in range(len(samples[used_hints]))])
                else:
                    hints.append([[] for _ in range(len(samples[used_hints]))])
            res_x, res_y, res_hints = oracle.oracle_fit(
                x_to_fit,
                y_to_fit,
                expr_idxs,
                hints,
                tree,
                name=oracle_name,
                use_parallel=True,
                use_seperate_type=params.oracle_seperation_type.split(","),
            )
            total_x = [
                *[
                    x[: params.max_input_points * max(1, params.max_number_bags)]
                    for x in x_to_fit
                ],
                *res_x,
            ]
            total_y = [
                *[
                    y[: params.max_input_points * max(1, params.max_number_bags)]
                    for y in y_to_fit
                ],
                *res_y,
            ]
            total_hints = [
                [*hint, *res_hint] for hint, res_hint in zip(hints, res_hints)
            ]
            time2 = time.time()
            try:
                dstr.fit(
                    total_x,
                    total_y,
                    total_hints,
                    verbose=verbose,
                    refinement_types=["id"],
                )
                best_gens_noref = copy.deepcopy(
                    dstr.retrieve_tree(
                        refinement_type="NoRef", dataset_idx=-1, with_infos=True
                    )
                )
            except:
                best_gens_noref = [
                    {
                        "refinement_type": "NoRef",
                        "predicted_tree": None,
                        "relabed_predicted_tree": None,
                        "time": 0,
                    }
                    for _ in range(len(total_y))
                ]
            time3 = time.time()
            best_gens = oracle.reverse(
                best_gens_noref[: len(x_to_fit)],
                best_gens_noref[len(x_to_fit) :],
                eliminate=True,
            )
            best_gens_e2e = copy.deepcopy(best_gens[1::2])
            best_gens_oracle = copy.deepcopy(best_gens[0::2])
            best_gens_oracle_mcts = copy.deepcopy(best_gens[0::2])
            best_gens_oracle_gp = copy.deepcopy(best_gens[0::2])
            time4 = time.time()
            for i, best_gen_oracle_mcts in enumerate(best_gens_oracle_mcts):
                num_variables = len(real_variables[i])
                base_grammar = {
                    "O": ["A"],
                    "A": [
                        "(A+A)",
                        "(A-A)",
                        "(A*A)",
                        "(A/A)",
                        "exp(A)",
                        "cos(B)",
                        "sin(B)",
                        "B",
                    ]
                    + [f"x_{i}" for i in range(num_variables)],
                    "B": ["(B+B)", "(B-B)", "1", "pi"]
                    + [f"(x_{i})**2" for i in range(num_variables)]
                    + [f"x_{i}" for i in range(num_variables)],
                }
                non_terminal = ["O", "A", "B"]
                mcts = MCTS(
                    params,
                    env,
                    base_grammar,
                    non_terminal,
                    mcts_print_freq=params.mcts_print_freq,
                    search_round=params.search_round,
                    num_simulations=params.num_simulations,
                    mcts_search_type="max",
                    early_stop=params.mcts_early_stop,
                )
                mcts.extract_subexpr(
                    best_gen_oracle_mcts["predicted_tree"], update=True
                )
                target_str, target_length = mcts.extract_target_str(
                    best_gen_oracle_mcts["predicted_tree"]
                )
                solutions = mcts.extract_original_solution(
                    best_gen_oracle_mcts["predicted_tree"], (x_to_fit[i], y_to_fit[i])
                )
                for _str, _len in zip(target_str, target_length):
                    if len(solutions) > 0:
                        mse = mcts.mse(solutions[-1][1], (x_to_fit[i], y_to_fit[i]))
                        if solutions[-1][2] > params.mcts_early_stop or mse < 1e-10:
                            break
                    productions = [["O", _str]]
                    nt = [c for c in _str if c == "A" or c == "B"]
                    solution = mcts.search(
                        (x_to_fit[i], y_to_fit[i]),
                        productions,
                        nt,
                        verbose=verbose,
                        max_time=max(5, 25 - _len),
                    )
                    solutions.append(solution)
                solutions.sort(key=lambda x: x[2])
                expr_str = solutions[-1][1]
                try:
                    node, _ = env.generator.infix_to_node(
                        expr_str.replace("pow", "**"),
                        variables=real_variables[i],
                        sp_parse=True,
                        allow_pow=True,
                        label_units=False,
                    )
                except:
                    node, _ = env.generator.infix_to_node(
                        "0",
                        variables=real_variables[i],
                        sp_parse=False,
                        label_units=False,
                    )
                best_gen_oracle_mcts["predicted_tree"] = node
            time5 = time.time()
            oracle_exprs = [e["predicted_tree"] for e in oracle.oracle_exprs] + [
                e["predicted_tree"] for e in best_gens_oracle
            ]
            for i in range(len(oracle_exprs)):
                oracle_exprs[i] = str(oracle_exprs[i])
                for k, v in symbolic_str_dic.items():
                    oracle_exprs[i] = oracle_exprs[i].replace(k, v)
                if "nan" in oracle_exprs[i] or "None" in oracle_exprs[i]:
                    oracle_exprs[i] = "0"
            gp_exprs = [
                (
                    oracle_exprs[v[0] : v[-1] + 1]
                    + [oracle_exprs[len(oracle.oracle_exprs) + i]]
                )
                for i, (k, v) in enumerate(
                    oracle.original_expr_idx_to_oracle_expr_idx.items()
                )
            ]
            for i, (best_gen_oracle_gp, gp_expr) in enumerate(
                zip(best_gens_oracle_gp, gp_exprs)
            ):
                if best_gen_oracle_gp["_mse"] < 1e-08:
                    continue
                num_variables = len(real_variables[i])
                assert (
                    num_variables == x_to_fit[i].shape[1]
                ), f"{num_variables}, {x_to_fit[i].shape[1]}"
                treeGenerator = TreeGenerator(params)
                gp = GeneticProgramming(treeGenerator, params, oracle, max_attemp=5)
                best_of_all = gp.run(
                    env,
                    num_variables,
                    (x_to_fit[i], y_to_fit[i]),
                    exprs=gp_expr,
                    verbose=0,
                )
                node = best_of_all.best()
                best_gen_oracle_gp["predicted_tree"] = node
            time6 = time.time()
            for k, v in infos.items():
                infos[k] = v.tolist()
            oracle_acc = oracle.evaluate_oracle_accuracy(tree)
            for _type, gens in zip(
                ["e2e", "oracle", "oraclemcts", "oraclegp"],
                [
                    best_gens_e2e,
                    best_gens_oracle,
                    best_gens_oracle_mcts,
                    best_gens_oracle_gp,
                ],
            ):
                predicted_tree = [best_gen["predicted_tree"] for best_gen in gens]
                batch_results["predicted_tree"].extend(
                    [str(_tree) for _tree in predicted_tree]
                )
                batch_results["tree"].extend([str(_tree) for _tree in tree])
                sym_acc = cal_sym_acc(
                    [str(_tree) for _tree in predicted_tree],
                    [str(_tree) for _tree in tree],
                )
                batch_results["sym_acc"].extend(sym_acc)
                y_tilde_to_fit = dstr.predict(
                    x_to_fit,
                    oracle_results=True,
                    oracle_tree=predicted_tree,
                    merged_type=oracle.merged_types,
                )
                assert len(y_to_fit) == len(y_tilde_to_fit)
                results_fit = compute_metrics(
                    {
                        "true": y_to_fit,
                        "predicted": y_tilde_to_fit,
                        "tree": tree,
                        "predicted_tree": predicted_tree,
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    batch_results[k + "_fit"].extend([round(_v, 4) for _v in v])
                del results_fit
                if self.params.prediction_sigmas is None:
                    prediction_sigmas = []
                else:
                    prediction_sigmas = [
                        float(sigma)
                        for sigma in self.params.prediction_sigmas.split(",")
                    ]
                for sigma in prediction_sigmas:
                    x_to_predict = samples["x_to_predict_{}".format(sigma)]
                    y_to_predict = samples["y_to_predict_{}".format(sigma)]
                    y_tilde_to_predict = dstr.predict(
                        x_to_predict,
                        oracle_results=True,
                        oracle_tree=predicted_tree,
                        merged_type=oracle.merged_types,
                    )
                    results_predict = compute_metrics(
                        {
                            "true": y_to_predict,
                            "predicted": y_tilde_to_predict,
                            "tree": tree,
                            "predicted_tree": predicted_tree,
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_predict.items():
                        batch_results[k + "_predict_{}".format(sigma)].extend(
                            [round(_v, 4) for _v in v]
                        )
                    del results_predict
                for t1, t2, v in zip(tree, predicted_tree, real_variables):
                    transition_dic = {f"x_{i}": _v for i, _v in enumerate(v)}
                    transition_dic.update(symbolic_str_dic)
                    if t1 is not None:
                        node1 = t1.copy()
                        self.env.generator.apply_transition(node1, transition_dic)
                        infix = str(node1)
                        batch_results["tree_labeled"].append(infix)
                    else:
                        batch_results["tree_labeled"].append(None)
                    if t2 is not None:
                        node2 = t2.copy()
                        self.env.generator.apply_transition(node2, transition_dic)
                        infix = str(node2)
                        batch_results["predicted_tree_labeled"].append(infix)
                    else:
                        batch_results["predicted_tree_labeled"].append(None)
                for _tree, _unit in zip(predicted_tree, samples["units"]):
                    try:
                        env.generator.label_units(_tree, units=_unit[:-1])
                        units_pred = env.equation_encoder.check_units(_tree, _unit)
                    except:
                        import traceback

                        message = traceback.format_exc()
                        units_pred = False
                    batch_results["units_pred"].append(units_pred)
                if _type == "e2e":
                    batch_results["elapsed_time"].extend(
                        [round((time3 - time2) / len(x_to_fit), 4)] * len(x_to_fit)
                    )
                elif _type == "oracle":
                    batch_results["elapsed_time"].extend(
                        [round((time4 - time1) / len(x_to_fit), 4)] * len(x_to_fit)
                    )
                elif _type == "oraclemcts":
                    batch_results["elapsed_time"].extend(
                        [round((time5 - time1) / len(x_to_fit), 4)] * len(x_to_fit)
                    )
                elif _type == "oraclegp":
                    batch_results["elapsed_time"].extend(
                        [round((time4 - time1 + time6 - time5) / len(x_to_fit), 4)]
                        * len(x_to_fit)
                    )
                for k in params.use_hints.split(","):
                    v = samples[k]
                    if k == "units":
                        batch_results["hints_" + k].extend(
                            [[__v.tolist() for __v in _v] for _v in v]
                        )
                batch_results["type"].extend([_type] * len(gens))
                batch_results["oracle_message"].extend([g["message"] for g in gens])
                if "oracle" in _type:
                    for k in oracle_acc[0].keys():
                        batch_results[
                            "oracle_result_{}".format(k.replace(",", "_"))
                        ].extend([_oracle_acc[k] for _oracle_acc in oracle_acc])
                else:
                    for k in oracle_acc[0].keys():
                        batch_results[
                            "oracle_result_{}".format(k.replace(",", "_"))
                        ].extend([[] for _ in range(len(gens))])
                batch_results = pd.DataFrame.from_dict(batch_results)
                if first_write:
                    os.makedirs(os.path.dirname(save_file), exist_ok=True)
                    batch_results.to_csv(save_file, index=False)
                else:
                    batch_results.to_csv(save_file, mode="a", header=False, index=False)
                first_write = False
                batch_results = defaultdict(list)
            bs = len(x_to_fit)
            pbar.update(bs)
        df = pd.read_csv(save_file, na_filter=True)
        df = df.fillna(0)
        for _type in ["e2e", "oracle", "oraclemcts", "oraclegp"]:
            df_current = df[df["type"] == _type]
            scores = {}
            print_columns = [
                f"r2_{b}"
                for b in [
                    "fit",
                    "predict_1.0",
                    "predict_2.0",
                    "predict_4.0",
                    "predict_8.0",
                    "predict_16.0",
                ]
            ]
            for k in print_columns:
                v = (np.array(list(df_current[k])) > 0.99).mean()
                kk = f"r2>0.99{k[2:]}"
                scores[kk] = v
                v = (np.array(list(df_current[k])) > 0.999).mean()
                kk = f"r2>0.999{k[2:]}"
                scores[kk] = v
            print_columns = [
                f"{a}_{b}"
                for a in [
                    "r2_zero",
                    "r2",
                    "accuracy_l1_biggio",
                    "accuracy_l1_1e-3",
                    "accuracy_l1_1e-2",
                    "accuracy_l1_1e-1",
                ]
                for b in [
                    "fit",
                    "predict_1.0",
                    "predict_2.0",
                    "predict_4.0",
                    "predict_8.0",
                    "predict_16.0",
                ]
            ] + ["_complexity_fit", "units_pred", "elapsed_time", "sym_acc"]
            for k in print_columns:
                v = df_current[k].mean()
                scores[k] = v
            if verbose_res:
                print_items = [
                    "r2>0.99",
                    "r2>0.999",
                    "r2_zero",
                    "accuracy_l1_biggio",
                    "accuracy_l1_1e-3",
                    "accuracy_l1_1e-2",
                    "accuracy_l1_1e-1",
                ]
                res_items = [
                    "fit",
                    "predict_1.0",
                    "predict_2.0",
                    "predict_4.0",
                    "predict_8.0",
                    "predict_16.0",
                ]
                print(f"type:{_type}")
                for prt_itm in print_items:
                    _res = [scores[f"{prt_itm}_{res_item}"] for res_item in res_items]
                    print(
                        f"{prt_itm:20s}: {_res[0]:.4f}, {_res[1]:.4f}, {_res[2]:.4f}, {_res[3]:.4f}, {_res[4]:.4f}, {_res[5]:.4f}"
                    )
                for k, v in [
                    ("complexity", "_complexity_fit"),
                    ("units_pred", "units_pred"),
                    ("elapsed_time", "elapsed_time"),
                    ("sym_acc", "sym_acc"),
                ]:
                    print(f"{k:20s}: {scores[v]:.4f}")
                print()
        return scores
