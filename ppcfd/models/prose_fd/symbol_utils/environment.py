from logging import getLogger

import numpy as np
import paddle

logger = getLogger()
from ppcfd.models.prose_fd.symbol_utils import generators
SPECIAL_WORDS = ["<BOS>", "<EOS>", "<PAD>", "<PLACEHOLDER>"]
SKIP_ITEM = "SKIP_ITEM"


class SymbolicEnvironment:
    def __init__(self, params):
        self.params = params
        self.rng = None
        self.seed = None
        self.float_precision = params.float_precision
        self.mantissa_len = params.mantissa_len
        self.max_size = None
        self.output_dim = params.max_output_dimension
        self.float_tolerance = 10**-params.float_precision
        self.additional_tolerance = [
            (10**-i) for i in range(params.float_precision + 1)
        ]
        assert (
            params.float_precision + 1
        ) % params.mantissa_len == 0, "Bad precision/mantissa len ratio"
        self.generator = generators.RandomFunctions(params, SPECIAL_WORDS)
        self.float_encoder = self.generator.float_encoder
        self.float_words = self.generator.float_words
        self.equation_encoder = self.generator.equation_encoder
        self.equation_words = self.generator.equation_words
        self.equation_words += self.float_words
        self.float_id2word = {i: s for i, s in enumerate(self.float_words)}
        self.equation_id2word = {i: s for i, s in enumerate(self.equation_words)}
        self.float_word2id = {s: i for i, s in self.float_id2word.items()}
        self.equation_word2id = {s: i for i, s in self.equation_id2word.items()}
        assert len(self.float_words) == len(set(self.float_words))
        assert len(self.equation_word2id) == len(set(self.equation_word2id))
        self.n_words = params.n_words = len(self.equation_words)
        logger.info(
            f"vocabulary: {len(self.float_word2id)} float words, {len(self.equation_word2id)} equation words"
        )
        self.bos_index = self.equation_word2id["<BOS>"]
        self.eos_index = self.equation_word2id["<EOS>"]
        self.pad_index = self.equation_word2id["<PAD>"]

    def word_to_idx(self, words, float_input=True):
        if float_input:
            return [
                [
                    paddle.LongTensor([self.float_word2id[dim] for dim in point])
                    for point in seq
                ]
                for seq in words
            ]
        else:
            return [
                paddle.LongTensor([self.equation_word2id[w] for w in eq])
                for eq in words
            ]

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            m = self.equation_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def idx_to_infix(self, lst, is_float=True, str_array=True):
        if is_float:
            idx_to_words = [self.float_id2word[int(i)] for i in lst]
        else:
            idx_to_words = [self.equation_id2word[int(term)] for term in lst]
        return self.word_to_infix(idx_to_words, is_float, str_array)

    """ # Not needed, only kept for reference

    def get_tx_grid(self, t_grid, x_grid, space_dim, x_grid_size):
        ""\"
        Generate tx_grid based on spacial and temporal grids
        Inputs:
            t_grid       (t_num, )
            x_grid       (x_num, ..., x_num, space_dim)
        Outputs:
            tx_grid      (t_num*x_grid_size, space_dim + 1)
        ""\"

        t_num = t_grid.size(0)

        x_grid = torch.reshape(x_grid, (1, x_grid_size, space_dim))
        t_grid = torch.reshape(t_grid, (t_num, 1))

        tx_grid = torch.zeros(t_num, x_grid_size, space_dim + 1, dtype=t_grid.dtype)

        tx_grid[:, :, 0] = t_grid.expand(t_num, x_grid_size)
        tx_grid[:, :, 1:] = x_grid.expand(t_num, x_grid_size, space_dim)

        return torch.reshape(tx_grid, (t_num * x_grid_size, space_dim + 1))

    def batch_data_pde_operator(self, data, t_eval, x_grid_size, step=1, dims=None):
        ""\"
        Batch list of data into a Tensor ready for operator data decoder
        Inputs:
            data           (bs, t_num, x_grid_size, dim)
        Outputs:
            data_input     (input_len, bs, 1+output_dim*x_grid_size)
            data_label     (data_len=t_num, bs, x_grid_size, output_dim)
            lengths        (bs, )
            dims           (bs, )
        ""\"
        length = t_eval.size(0)
        input_len = self.params.input_len // step
        lengths = torch.LongTensor([input_len for _ in data])
        if dims is None:
            dims = torch.LongTensor([eq.size(-1) for eq in data])
        else:
            dims = torch.LongTensor(dims)

        data_label = torch.stack(data).transpose(0, 1)  # (data_len, bs, x_grid_size, output_dim)

        reshape_dim = x_grid_size * data_label.size(-1)
        bs = len(dims)
        t_input = t_eval[0 : self.params.input_len : step][:, None, None]  # (data_len, 1, 1)
        t_input = t_input.expand(input_len, bs, 1)
        data_input = data_label[0 : self.params.input_len : step].reshape(input_len, bs, reshape_dim)
        data_input = torch.cat((t_input, data_input), dim=-1)

        return data_input, data_label, lengths, dims

    def batch_equations(self, equations):
        ""\"
        Batch list of text seq into a Tensor ready for text decoder
        Input:
            equations  list of n sequences (torch.LongTensor vectors)
        Outputs:
            sent       Tensor (slen, n) where slen is the length of the longest sentence
            lengths    (slen, ) length of each sentence
        ""\"
        lengths = torch.LongTensor([2 + len(eq) for eq in equations])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.float_word2id["<PAD>"])
        sent[0] = self.equation_word2id["<BOS>"]
        for i, eq in enumerate(equations):
            sent[1 : lengths[i] - 1, i].copy_(eq)
            sent[lengths[i] - 1, i] = self.equation_word2id["<EOS>"]
        return sent, lengths

    def batch_equations_placeholder(self, equations):
        ""\"
        Batch list of text seq into a Tensor ready for text encoder (no <BOS> or <EOS>)
        Input:
            equations  list of n sequences (torch.LongTensor vectors)
        Outputs:
            sent       Tensor (slen, n) where slen is the length of the longest sentence
            lengths    (slen, ) length of each sentence
        ""\"
        lengths = torch.LongTensor([len(eq) for eq in equations])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.float_word2id["<PAD>"])
        for i, eq in enumerate(equations):
            sent[: lengths[i], i].copy_(eq)
        return sent, lengths
    """


if __name__ == "__main__":
    import logging
    import sys

    import hydra

    replace = {"add": "+", "sub": "-", "mul": "*", "neg": "-"}

    def replace_ops(s):
        for k, v in replace.items():
            s = s.replace(k, v)
        return s

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    @hydra.main(version_base=None, config_path="../configs", config_name="main")
    def test(params):
        symbol_env = SymbolicEnvironment(params.symbol)
        type_label = "com_ns"
        tree = symbol_env.generator.get_tree(type_label, {"F": 0.2})
        tree_str = replace_ops(str(tree))
        print(tree_str)
        print()
        tree_prefix = tree.prefix()
        print(tree_prefix, len(tree_prefix.split(",")))
        for w in symbol_env.equation_encoder.encode(tree):
            assert w in symbol_env.equation_word2id, w

    test()
