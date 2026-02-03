from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Tuple

import numpy as np
import paddle

from ...paddle_utils import *
from ..utils import to_cuda

MultiDimensionalFloat = List[float]
XYPair = Tuple[MultiDimensionalFloat, MultiDimensionalFloat]
Sequence = List[XYPair]


class Embedder(ABC, paddle.nn.Module):
    """
    Base class for embedders, transforms a sequence of pairs into a sequence of embeddings.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, sequences: List[Sequence]) -> Tuple[paddle.Tensor, paddle.Tensor]:
        pass

    @abstractmethod
    def num_encode(self, sequences: List[Sequence]) -> List[paddle.Tensor]:
        pass

    def batch(self, seqs: List[paddle.Tensor]) -> Tuple[paddle.Tensor, paddle.Tensor]:
        raise NotImplementedError

    def embed(self, batch: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_length_after_batching(self, sequences: List[Sequence]) -> List[int]:
        pass


class LinearPointEmbedder(Embedder):
    def __init__(self, params, env):
        from .transformer import Embedding

        super().__init__()
        self.env = env
        self.params = params
        self.input_dim = params.emb_emb_dim
        self.output_dim = params.enc_emb_dim
        self.embeddings = Embedding(
            len(self.env.float_id2word),
            self.input_dim,
            padding_idx=env.float_word2id["<PAD>"],
        )
        self.float_scalar_descriptor_len = 2 + self.params.mantissa_len
        self.total_dimension = (
            self.params.max_input_dimension + self.params.max_output_dimension
        )
        self.float_vector_descriptor_len = (
            self.float_scalar_descriptor_len * self.total_dimension
        )
        self.activation_fn = paddle.nn.functional.relu
        size = (self.float_vector_descriptor_len + 2) * self.input_dim
        hidden_size = size * self.params.emb_expansion_factor
        self.hidden_layers = paddle.nn.ModuleList()
        self.hidden_layers.append(paddle.compat.nn.Linear(size, hidden_size))
        for i in range(self.params.n_emb_layers - 1):
            self.hidden_layers.append(paddle.compat.nn.Linear(hidden_size, hidden_size))
        self.fc = paddle.compat.nn.Linear(hidden_size, self.output_dim)
        self.max_seq_len = self.params.max_len

        # 预计算常用token ID以减少字典查询
        self.common_token_ids = {
            "<DATA_POINT>": self.env.float_word2id["<DATA_POINT>"],
            "</DATA_POINT>": self.env.float_word2id["</DATA_POINT>"],
            "<INPUT_PAD>": self.env.float_word2id["<INPUT_PAD>"],
            "<OUTPUT_PAD>": self.env.float_word2id["<OUTPUT_PAD>"],
            "<HINT_PAD>": self.env.float_word2id["<HINT_PAD>"],
            "<PHYSICAL_UNITS>": self.env.float_word2id["<PHYSICAL_UNITS>"],
            "</PHYSICAL_UNITS>": self.env.float_word2id["</PHYSICAL_UNITS>"],
            "<COMPLEXITY>": self.env.float_word2id["<COMPLEXITY>"],
            "</COMPLEXITY>": self.env.float_word2id["</COMPLEXITY>"],
            "<UNKNOWN_COMPLEXITY>": self.env.float_word2id["<UNKNOWN_COMPLEXITY>"],
            "<UNARY>": self.env.float_word2id["<UNARY>"],
            "</UNARY>": self.env.float_word2id["</UNARY>"],
            "<ADD_STRUCTURE>": self.env.float_word2id["<ADD_STRUCTURE>"],
            "</ADD_STRUCTURE>": self.env.float_word2id["</ADD_STRUCTURE>"],
            "<MUL_STRUCTURE>": self.env.float_word2id["<MUL_STRUCTURE>"],
            "</MUL_STRUCTURE>": self.env.float_word2id["</MUL_STRUCTURE>"],
            "<USED_CONST>": self.env.float_word2id["<USED_CONST>"],
            "</USED_CONST>": self.env.float_word2id["</USED_CONST>"],
        }

        # 优化1: 预生成填充模板 (避免每次重新生成列表)
        max_input_pad = self.params.max_input_dimension * self.float_scalar_descriptor_len
        max_output_pad = self.params.max_output_dimension * self.float_scalar_descriptor_len

        self.input_pad_template = ["<INPUT_PAD>"] * max_input_pad
        self.output_pad_template = ["<OUTPUT_PAD>"] * max_output_pad

        # 优化2: 预生成填充ID模板 (避免重复查询字典)
        self.input_pad_ids = [self.env.float_word2id["<INPUT_PAD>"]] * max_input_pad
        self.output_pad_ids = [self.env.float_word2id["<OUTPUT_PAD>"]] * max_output_pad

    def compress(
        self, sequences_embeddings: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(2+mantissa_len), B, d) tensors
        Returns: (N_max, B, d)
        """
        max_len, bs, float_descriptor_length, dim = sequences_embeddings.size()
        sequences_embeddings = sequences_embeddings.view(max_len, bs, -1)
        for layer in self.hidden_layers:
            sequences_embeddings = self.activation_fn(layer(sequences_embeddings))
        sequences_embeddings = self.fc(sequences_embeddings)
        return sequences_embeddings

    def forward(
        self, sequences: List[Sequence], hints
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        sequences = self.num_encode(sequences)
        if self.params.use_hints:
            hints = self.hint_encode(hints, self.params.use_hints)
            sequences = [
                paddle.cat((hint, sequence), dim=0)
                for hint, sequence in zip(hints, sequences)
            ]
        sequences, sequences_len = self.batch(sequences)
        sequences, sequences_len = to_cuda(
            sequences,
            sequences_len,
            use_cpu=self.fc.weight.device.type == "cpu",
            device=self.env.params.device,
        )
        sequences_embeddings = self.embed(sequences)
        sequences_embeddings = self.compress(sequences_embeddings)
        return sequences_embeddings, sequences_len

    def num_encode(self, sequences: List[Sequence]) -> List[paddle.Tensor]:
        """优化的数值编码方法 - 使用批量编码 + 填充模板 + 嵌套循环优化"""
        res = []
        for seq in sequences:
            if len(seq) == 0:
                res.append(paddle.to_tensor([], dtype="int64"))
                continue

            # 收集所有x和y值
            x_values = []
            y_values = []
            for x, y in seq:
                x_values.append(x)
                y_values.append(y)

            # 转换为NumPy数组进行批量编码
            x_batch = np.array(x_values)  # shape: (n_points, n_vars)
            y_batch = np.array(y_values)  # shape: (n_points, 1) 或 (n_points,)

            # 确保x_batch是2D
            if len(x_batch.shape) == 1:
                x_batch = x_batch.reshape(-1, 1)

            # 确保y_batch是2D
            if len(y_batch.shape) == 1:
                y_batch = y_batch.reshape(-1, 1)

            # 获取变量数量
            n_vars = x_batch.shape[1]

            # 验证输入维度不超过最大值
            if n_vars > self.params.max_input_dimension:
                raise ValueError(
                    f"输入维度 {n_vars} 超过最大允许维度 {self.params.max_input_dimension}"
                )

            # 批量编码
            x_encoded_batch = self.env.float_encoder.encode_batch(x_batch)
            y_encoded_batch = self.env.float_encoder.encode_batch(y_batch)

            # 构建序列
            seq_toks = []
            n_points = len(seq)
            n_vars = x_batch.shape[1]

            # 计算填充长度（使用max确保非负）
            input_pad_count = max(0, (self.params.max_input_dimension - n_vars) * self.float_scalar_descriptor_len)
            output_pad_count = max(0, (self.params.max_output_dimension - 1) * self.float_scalar_descriptor_len)

            for i in range(n_points):
                # 优化3: 使用chain.from_iterable减少嵌套循环
                start_idx = i * n_vars
                end_idx = start_idx + n_vars
                x_toks = list(chain.from_iterable(x_encoded_batch[start_idx:end_idx]))

                y_toks = y_encoded_batch[i]

                # 优化1: 使用预生成的填充模板
                x_toks_padded = x_toks + self.input_pad_template[:input_pad_count]
                y_toks_padded = y_toks + self.output_pad_template[:output_pad_count]

                # 优化2: 使用预计算的ID + 预生成的填充ID
                toks_ids = [self.common_token_ids["<DATA_POINT>"]]
                
                # 对于x_toks中的非填充部分,仍需查询
                toks_ids.extend([self.env.float_word2id[tok] for tok in x_toks])
                # 添加填充ID
                toks_ids.extend(self.input_pad_ids[:input_pad_count])
                
                # 对于y_toks中的非填充部分,仍需查询
                toks_ids.extend([self.env.float_word2id[tok] for tok in y_toks])
                # 添加填充ID
                toks_ids.extend(self.output_pad_ids[:output_pad_count])
                
                toks_ids.append(self.common_token_ids["</DATA_POINT>"])

                seq_toks.append(toks_ids)

            res.append(paddle.to_tensor(seq_toks, dtype="int64"))
        return res

    def batch(self, seqs: List[paddle.Tensor]) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """优化的批处理方法 - 使用paddle.full预分配"""
        pad_id = self.env.float_word2id["<PAD>"]
        lengths = [len(x) for x in seqs]
        bs, slen = len(lengths), max(lengths)

        # 使用paddle.full替代fill_操作
        sent = paddle.full(
            shape=[slen, bs, self.float_vector_descriptor_len + 2],
            fill_value=pad_id,
            dtype="int64",
        )

        # 批量赋值
        for i, seq in enumerate(seqs):
            if len(seq) > 0:
                sent[0 : len(seq), i, :] = seq

        return sent, paddle.to_tensor(lengths, dtype="int64")

    def embed(self, batch: paddle.Tensor) -> paddle.Tensor:
        return self.embeddings(batch)

    def get_length_after_batching(self, seqs: List[Sequence]) -> paddle.Tensor:
        """
        线程安全的序列长度计算方法

        针对多线程 DataLoader 环境优化：
        - 完全在 Python 层面处理，避免 GPU 多线程问题
        - 增强错误诊断，捕获并发数据问题
        - 线程安全：Python 列表操作是原子性的
        """
        # 1. 在 Python 层面计算长度（线程安全）
        try:
            length_values = [len(seq) for seq in seqs]
        except Exception as e:
            print(f"[ERROR] Failed to compute sequence lengths: {e}")
            print(f"  seqs type: {type(seqs)}")
            print(f"  seqs length: {len(seqs)}")
            if len(seqs) > 0:
                print(f"  first seq type: {type(seqs[0])}")
            raise

        # 2. 计算最大值（Python层面，避免 GPU 操作）
        if not length_values:
            max_length = 0
        else:
            max_length = max(length_values)

        # 3. 验证（增强诊断）
        if max_length > self.max_seq_len:
            print(f"[ERROR] Abnormal sequence length detected!")
            print(f"  max_length: {max_length}")
            print(f"  max_seq_len: {self.max_seq_len}")
            print(f"  length_values: {length_values}")
            print(f"  seqs count: {len(seqs)}")

            # 检查是否有异常值
            for i, length in enumerate(length_values):
                if length > self.max_seq_len:
                    print(f"  ❌ seq[{i}] has abnormal length: {length}")

            # 仍然抛出异常，但提供更多信息
            raise AssertionError(
                f"序列长度 {max_length} 超过最大限制 {self.max_seq_len}。"
                f"检测到异常数据，详见日志。"
            )

        # 4. 创建张量（会在当前设备，线程安全）
        lengths = paddle.to_tensor(length_values, dtype=paddle.long)

        return lengths

    def hint_encode(self, hints, use_hints):
        """优化的提示编码方法 - 减少字符串操作和字典查询"""
        use_hints_list = use_hints.split(",")
        res = []

        # 预计算填充长度和ID
        hint_pad_len = (
            self.params.max_input_dimension + self.params.max_output_dimension
        ) * self.float_scalar_descriptor_len
        hint_pad_id = self.env.float_word2id["<HINT_PAD>"]

        for seq_id in range(len(hints[0])):
            seq_toks = []
            current_hints_idx = 0

            if "units" in use_hints_list:
                units = hints[current_hints_idx]
                for i, unit in enumerate(units[seq_id]):
                    un = self.env.equation_encoder.units_encode(unit)
                    var_name = f"x_{i}" if i != len(units[seq_id]) - 1 else "y"

                    # 直接构建ID列表,减少字符串操作
                    units_toks = [
                        self.common_token_ids["<PHYSICAL_UNITS>"],
                        self.env.float_word2id[var_name],
                    ]
                    units_toks.extend([self.env.float_word2id[u] for u in un])
                    units_toks.append(self.common_token_ids["</PHYSICAL_UNITS>"])

                    # 添加填充
                    pad_len = hint_pad_len - len(un) - 1
                    units_toks.extend([hint_pad_id] * pad_len)

                    seq_toks.append(units_toks)
                current_hints_idx += 1

            if "complexity" in use_hints_list:
                complexity = hints[current_hints_idx]
                for c in complexity[seq_id]:
                    # 复杂度可以是字符串(simple/middle/hard)或数字
                    if isinstance(c, str):
                        com_tok = f"COMPLEXITY:{c}"
                    elif c != 0:
                        com_tok = f"COMPLEXITY:{c}"
                    else:
                        com_tok = "<UNKNOWN_COMPLEXITY>"

                    com_toks = [
                        self.common_token_ids["<COMPLEXITY>"],
                        self.env.float_word2id[com_tok],
                        self.common_token_ids["</COMPLEXITY>"],
                    ]
                    # 添加填充
                    pad_len = hint_pad_len - 1
                    com_toks.extend([hint_pad_id] * pad_len)

                    seq_toks.append(com_toks)
                current_hints_idx += 1

            if "unarys" in use_hints_list:
                unarys = hints[current_hints_idx]
                unary_toks = [self.common_token_ids["<UNARY>"]]
                unary_toks.extend([self.env.float_word2id[u] for u in unarys[seq_id]])
                unary_toks.append(self.common_token_ids["</UNARY>"])

                # 添加填充
                pad_len = hint_pad_len - len(unarys[seq_id])
                unary_toks.extend([hint_pad_id] * pad_len)

                seq_toks.append(unary_toks)
                current_hints_idx += 1

            if "add_structure" in use_hints_list:
                add_structure = hints[current_hints_idx]
                for a in add_structure[seq_id]:
                    add_toks = [self.common_token_ids["<ADD_STRUCTURE>"]]
                    add_toks.extend([self.env.float_word2id[f"x_{j}"] for j in a])
                    add_toks.append(self.common_token_ids["</ADD_STRUCTURE>"])

                    # 添加填充
                    pad_len = hint_pad_len - len(a)
                    add_toks.extend([hint_pad_id] * pad_len)

                    seq_toks.append(add_toks)
                current_hints_idx += 1

            if "mul_structure" in use_hints_list:
                mul_structure = hints[current_hints_idx]
                for m in mul_structure[seq_id]:
                    mul_toks = [self.common_token_ids["<MUL_STRUCTURE>"]]
                    mul_toks.extend([self.env.float_word2id[f"x_{j}"] for j in m])
                    mul_toks.append(self.common_token_ids["</MUL_STRUCTURE>"])

                    # 添加填充
                    pad_len = hint_pad_len - len(m)
                    mul_toks.extend([hint_pad_id] * pad_len)

                    seq_toks.append(mul_toks)
                current_hints_idx += 1

            if "consts" in use_hints_list:
                consts = hints[current_hints_idx]
                for _value, _units in consts[seq_id]:
                    if _value != "pi":
                        const_toks_content = [
                            *self.env.float_encoder.encode(np.array([_value])),
                            *self.env.equation_encoder.units_encode(_units),
                        ]
                    else:
                        const_toks_content = [
                            "pi",
                            *self.env.equation_encoder.units_encode(_units),
                        ]

                    const_toks = [self.common_token_ids["<USED_CONST>"]]
                    const_toks.extend(
                        [self.env.float_word2id[tok] for tok in const_toks_content]
                    )
                    const_toks.append(self.common_token_ids["</USED_CONST>"])

                    # 添加填充
                    pad_len = hint_pad_len - len(const_toks_content)
                    const_toks.extend([hint_pad_id] * pad_len)

                    seq_toks.append(const_toks)
                current_hints_idx += 1

            res.append(paddle.to_tensor(seq_toks, dtype="int64"))
        return res
