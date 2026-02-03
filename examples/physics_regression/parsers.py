import argparse

import numpy as np

from ppcfd.models.physicsregression._paths import FEYNMAN_EQUATIONS_PATH
from ppcfd.models.physicsregression._paths import UNITS_CSV_PATH
from ppcfd.models.physicsregression._paths import example_data_path
from ppcfd.models.physicsregression.symbolicregression.envs import ENVS
from ppcfd.models.physicsregression.symbolicregression.utils import bool_flag


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="Function prediction")
    parser.add_argument(
        "--dump_path", type=str, default="", help="Experiment dump path"
    )
    parser.add_argument(
        "--refinements_types",
        type=str,
        default="method=BFGS_batchsize=256_metric=/_mse",
        help="What refinement to use. Should separate by _ each arg and value by =. None does not do any refinement",
    )
    parser.add_argument(
        "--eval_dump_path", type=str, default=None, help="Evaluation dump path"
    )
    parser.add_argument(
        "--save_results", type=bool, default=True, help="Should we save results?"
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Print every n steps"
    )
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument(
        "--fp16", type=bool_flag, default=False, help="Run model with float16"
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=-1,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )
    parser.add_argument(
        "--rescale", type=int, default=1, help="Whether to rescale at inference."
    )
    parser.add_argument(
        "--embedder_type",
        type=str,
        default="LinearPoint",
        help="[TNet, LinearPoint, Flat, AttentionPoint] How to pre-process sequences before passing to a transformer.",
    )
    parser.add_argument(
        "--emb_emb_dim", type=int, default=64, help="Embedder embedding layer size"
    )
    parser.add_argument(
        "--enc_emb_dim", type=int, default=512, help="Encoder embedding layer size"
    )
    parser.add_argument(
        "--dec_emb_dim", type=int, default=512, help="Decoder embedding layer size"
    )
    parser.add_argument(
        "--n_emb_layers", type=int, default=1, help="Number of layers in the embedder"
    )
    parser.add_argument(
        "--n_enc_layers",
        type=int,
        default=2,
        help="Number of Transformer layers in the encoder",
    )
    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=16,
        help="Number of Transformer layers in the decoder",
    )
    parser.add_argument(
        "--n_enc_heads",
        type=int,
        default=16,
        help="Number of Transformer encoder heads",
    )
    parser.add_argument(
        "--n_dec_heads",
        type=int,
        default=16,
        help="Number of Transformer decoder heads",
    )
    parser.add_argument(
        "--emb_expansion_factor",
        type=int,
        default=1,
        help="Expansion factor for embedder",
    )
    parser.add_argument(
        "--n_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer encoder",
    )
    parser.add_argument(
        "--n_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer decoder",
    )
    parser.add_argument(
        "--norm_attention",
        type=bool_flag,
        default=False,
        help="Normalize attention and train temperaturee in Transformer",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )
    parser.add_argument(
        "--enc_positional_embeddings",
        type=str,
        default=None,
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--dec_positional_embeddings",
        type=str,
        default="learnable",
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=0,
        help="Base seed for environments (-1 to use timestamp seed)",
    )
    parser.add_argument(
        "--test_env_seed", type=int, default=1, help="Test seed for environments"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Number of sentences per batch"
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=64,
        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_inverse_sqrt,warmup_updates=10000",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.5,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--n_steps_per_epoch", type=int, default=3000, help="Number of steps per epoch"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Number of epochs"
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )
    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for DataLoader",
    )
    parser.add_argument(
        "--train_noise_gamma",
        type=float,
        default=0.0,
        help="Should we train with additional output noise",
    )
    parser.add_argument(
        "--ablation_to_keep", type=str, default=None, help="which ablation should we do"
    )
    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="split into chunks of size max_input_points at eval",
    )
    parser.add_argument(
        "--n_trees_to_refine", type=int, default=10, help="refine top n trees"
    )
    parser.add_argument(
        "--export_data",
        type=bool_flag,
        default=False,
        help="Export data and disable training.",
    )
    parser.add_argument(
        "--reload_data",
        type=str,
        default="",
        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)",
    )
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )
    parser.add_argument(
        "--batch_load",
        type=bool_flag,
        default=False,
        help="Load training set by batches (of size reload_size).",
    )
    parser.add_argument(
        "--env_name", type=str, default="functions", help="Environment name"
    )
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)
    parser.add_argument("--tasks", type=str, default="functions", help="Tasks")
    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=True,
        help="Evaluate with beam search decoding.",
    )
    parser.add_argument(
        "--max_generated_output_len",
        type=int,
        default=200,
        help="Max generated output length",
    )
    parser.add_argument(
        "--beam_eval_train",
        type=int,
        default=0,
        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size, default = 1 (greedy decoding)",
    )
    parser.add_argument(
        "--beam_type", type=str, default="sampling", help="Beam search or sampling"
    )
    parser.add_argument(
        "--beam_temperature",
        type=int,
        default=0.1,
        help="Beam temperature for sampling",
    )
    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )
    parser.add_argument("--beam_selection_metrics", type=int, default=1)
    parser.add_argument("--max_number_bags", type=int, default=-1)
    parser.add_argument(
        "--reload_model", type=str, default="", help="Reload a pretrained model"
    )
    parser.add_argument(
        "--reload_checkpoint", type=str, default="", help="Reload a checkpoint"
    )
    parser.add_argument(
        "--validation_metrics",
        type=str,
        default="r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity",
        help="What metrics should we report? accuracy_tolerance/_l1_error/r2/_complexity/_relative_complexity/is_symbolic_solution",
    )
    parser.add_argument(
        "--debug_train_statistics",
        type=bool,
        default=False,
        help="whether we should print infos distributions",
    )
    parser.add_argument(
        "--eval_noise_gamma",
        type=float,
        default=0.0,
        help="Should we evaluate with additional output noise",
    )
    parser.add_argument(
        "--eval_size", type=int, default=10000, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_noise_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Type of noise added at test time",
    )
    parser.add_argument(
        "--eval_noise", type=float, default=0, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_only", type=bool_flag, default=False, help="Only run evaluations"
    )
    parser.add_argument(
        "--eval_from_exp", type=str, default="", help="Path of experiment to use"
    )
    parser.add_argument(
        "--eval_data", type=str, default="", help="Path of data to eval"
    )
    parser.add_argument(
        "--eval_verbose", type=int, default=0, help="Export evaluation details"
    )
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )
    parser.add_argument(
        "--eval_input_length_modulo",
        type=int,
        default=-1,
        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation",
    )
    parser.add_argument("--eval_on_pmlb", type=bool, default=False)
    parser.add_argument("--eval_in_domain", type=bool, default=True)
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")
    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--windows",
        type=bool_flag,
        default=False,
        help="Windows version (no multiprocessing for eval)",
    )
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=0,
        help="Save the model periodically (0 to disable)",
    )
    parser.add_argument(
        "--save_periodic_from",
        type=int,
        default=0,
        help="Save the model periodically from which epoch?",
    )
    parser.add_argument("--expr_data", type=str, default=None)
    parser.add_argument(
        "--add_consts",
        type=int,
        default=1,
        help="whether to add const during sample, default to be 1(True)",
    )
    parser.add_argument(
        "--encode_expr_form",
        type=str,
        default="prefix",
        help="how to encode expr? prefix(default)/infix",
    )
    parser.add_argument(
        "--pre_differentiate_path",
        type=str,
        default=None,
        help="whether we have differentiate before to speed up",
    )
    parser.add_argument(
        "--expr_train_data_path",
        type=str,
        default=None,
        help="which dataest to use when sampling as training",
    )
    parser.add_argument(
        "--expr_valid_data_path",
        type=str,
        default=None,
        help="which dataest to use when sampling as validation",
    )
    parser.add_argument(
        "--expr_test_data_path",
        type=str,
        default=example_data_path("exprs_test_ranked.json"),
        help="which dataest to use when sampling as testing",
    )
    parser.add_argument(
        "--sub_expr_train_path",
        type=str,
        default=None,
        help="which sub-expr dataest to use when sampling as training",
    )
    parser.add_argument(
        "--sub_expr_valid_path",
        type=str,
        default=None,
        help="which sub-expr dataest to use when sampling as validation",
    )
    parser.add_argument(
        "--feynman_equations_path",
        type=str,
        default=str(FEYNMAN_EQUATIONS_PATH),
        help="Path to the bundled Feynman equations metadata file",
    )
    parser.add_argument(
        "--units_csv_path",
        type=str,
        default=str(UNITS_CSV_PATH),
        help="Path to the bundled physical units metadata file",
    )
    parser.add_argument(
        "--generate_datapoints_distribution",
        type=str,
        default="multi-random/uniform",
        help="which methods to generate datapoints: multi-random/uniform(original end2end), single-random/uniform",
    )
    parser.add_argument(
        "--random_variables_sequence",
        type=int,
        default=0,
        help="whether to shuffle variable sequence?",
    )
    parser.add_argument(
        "--p_add",
        type=float,
        default=0.15,
        help="probability to add random additive const",
    )
    parser.add_argument(
        "--p_mul",
        type=float,
        default=0.12,
        help="probability to add random multiplicative const",
    )
    parser.add_argument(
        "--use_exprs", type=int, default=-1, help="how many exprs do we use to train?"
    )
    parser.add_argument(
        "--prediction_sigmas",
        type=str,
        default="1,2,4,8,16",
        help="sigmas value for generation predicts",
    )
    parser.add_argument(
        "--sample_expr_num", type=int, default=1, help="how many same expr to sample?"
    )
    parser.add_argument(
        "--eval_feynman",
        type=int,
        default=0,
        help="whether to evaluate on feynman dataset",
    )
    parser.add_argument(
        "--eval_start_from",
        type=int,
        default=0,
        help="where to start at testing dataset",
    )
    parser.add_argument("--dim_length", type=int, default=5)
    parser.add_argument(
        "--max_len", type=int, default=200, help="Max number of terms in the series"
    )
    parser.add_argument(
        "--min_len_per_dim", type=int, default=5, help="Min number of terms per dim"
    )
    parser.add_argument(
        "--tokens_per_batch",
        type=int,
        default=10000,
        help="max number of tokens per batch",
    )
    parser.add_argument(
        "--use_hints",
        type=str,
        default="units,complexity,unarys,consts",
        help="which hints to use? optional:units/complexity/unarys/add_structure/mul_structure/consts, divided by ','",
    )
    parser.add_argument(
        "--sample_unary_hints",
        type=str,
        default=None,
        help="given specific unary hints to sample?",
    )
    parser.add_argument(
        "--sample_complexity_hints",
        type=str,
        default=None,
        help="given specific complexity hints to sample?",
    )
    parser.add_argument(
        "--sample_structure_hints",
        type=str,
        default=None,
        help="given specific structure hints to sample?",
    )
    parser.add_argument(
        "--sample_dimension_hints",
        type=str,
        default=None,
        help="given specific dimension hints to sample?",
    )
    parser.add_argument(
        "--sample_const_hints",
        type=str,
        default=None,
        help="given specific const hints to sample?",
    )
    parser.add_argument(
        "--use_dimension_mask",
        type=int,
        default=0,
        help="whether use a dimensional mask during inference to help select next word",
    )
    parser.add_argument(
        "--decode_physical_units",
        type=str,
        default=None,
        help="how to decode output physical units. (None, single-seq, double-seq)",
    )
    parser.add_argument(
        "--units_criterion_constrain",
        type=int,
        default=0,
        help="whether mask those predicted expr whose units contract with formal criterion",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--refinement_strategy",
        type=str,
        default="id,neg,inv,linear,safe,safe-neg,safe-inv",
        help="which methods do we take to refine formulas",
    )
    parser.add_argument(
        "--num_bfgs",
        type=int,
        default=10,
        help="number of times to restart bfgs at different init-value",
    )
    parser.add_argument(
        "--oracle_seperation_type",
        type=str,
        default="id,inv,arcsin,arccos,sqrt",
        help="which strategy do we need to take?",
    )
    parser.add_argument(
        "--oracle_sep_multinum",
        type=int,
        default=1,
        help="do we need to sample multiple seperation strategy",
    )
    parser.add_argument("--eps", type=int, default=1e-06, help="")
    parser.add_argument(
        "--D_max",
        type=int,
        default=10,
        help="the maximum dimension for function generator",
    )
    parser.add_argument(
        "--u_max",
        type=int,
        default=5,
        help="the maximum number of unary operator for function generator",
    )
    parser.add_argument(
        "--search_round",
        type=int,
        default=10000,
        help="how many searchs for a single mcts ",
    )
    parser.add_argument(
        "--mcts_print_freq",
        type=int,
        default=100,
        help="how often do we print current searching state",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=10,
        help="how many times of random simulations for init reward, using for MCTS.simulations",
    )
    parser.add_argument(
        "--UCT_const",
        type=float,
        default=1 / np.sqrt(2),
        help="coefficient of UCT, to parismony the visits",
    )
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.9999,
        help="the discount factor using for reward, reward = discount_factor ** complexity / (1 + RMSE)",
    )
    parser.add_argument(
        "--t_max", type=int, default=100, help="max depth for selection"
    )
    parser.add_argument(
        "--expr_max", type=int, default=50, help="for max length for simulation"
    )
    parser.add_argument(
        "--simulation_max",
        type=int,
        default=5,
        help="the max step for single simulation",
    )
    parser.add_argument(
        "--max_production_length",
        type=int,
        default=10,
        help="the max length for good module update",
    )
    parser.add_argument(
        "--greedy_eps", type=float, default=0.0, help="epsilon for greedy search"
    )
    parser.add_argument(
        "--mcts_early_stop",
        type=float,
        default=0.99,
        help="reward threshold for mcts early stop",
    )
    parser.add_argument(
        "--tornament_size",
        type=int,
        default=12,
        help="Number of individuals in a single tournament for GP.",
    )
    parser.add_argument(
        "--p_tornament",
        type=float,
        default=0.86,
        help="Probability for the winner of each tournament to be chosen in GP.",
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=15,
        help="Number of populations for GP and population construction.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=33,
        help="Number of individuals in each population for GP.",
    )
    parser.add_argument(
        "--mutation_epoch",
        type=int,
        default=550 * 3,
        help="Number of mutation events per epoch in GP.",
    )
    parser.add_argument(
        "--gp_epochs", type=int, default=40, help="Total number of epochs for GP."
    )
    parser.add_argument(
        "--max_complexity",
        type=int,
        default=30,
        help="Maximum allowable complexity in GP.",
    )
    parser.add_argument(
        "--alpha_temperature_scale",
        type=float,
        default=0.1,
        help="Alpha parameter for temperature scaling in regularized evolution.",
    )
    parser.add_argument(
        "--prob_negative_mutate_const",
        type=float,
        default=0.01,
        help="Probability of constant changing to negative during GP mutation.",
    )
    parser.add_argument(
        "--p_crossover",
        type=float,
        default=0.066,
        help="Probability of crossover in GP.",
    )
    parser.add_argument(
        "--optimize_probability",
        type=float,
        default=0.14,
        help="Probability of optimizing constants during GP.",
    )
    parser.add_argument(
        "--parsimony",
        type=float,
        default=0.0032,
        help="Parsimony coefficient for penalizing complexity in GP.",
    )
    parser.add_argument(
        "--fraction_replaced",
        type=float,
        default=0.00036,
        help="Fraction of individuals to be replaced during migration in GP.",
    )
    parser.add_argument(
        "--fraction_replaced_hof",
        type=float,
        default=0.035,
        help="Fraction of individuals to be replaced by hall of fame during migration in GP.",
    )
    parser.add_argument(
        "--use_recorder",
        type=bool_flag,
        default=False,
        help="Whether to use recorder for logging GP events.",
    )
    return parser
