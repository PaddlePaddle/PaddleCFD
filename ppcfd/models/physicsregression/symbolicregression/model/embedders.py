from abc import ABC, abstractmethod
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
        self.hidden_layers.append(TraceableLinear(size, hidden_size))
        for i in range(self.params.n_emb_layers - 1):
            self.hidden_layers.append(TraceableLinear(hidden_size, hidden_size))
        self.fc = TraceableLinear(hidden_size, self.output_dim)
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

        self._build_float_id_tables()

    def _build_float_id_tables(self):
        """预计算 float token 的 id 查表，供向量化的 `num_encode` 使用。

        `FloatSequences.encode` 把一个浮点数编码成 mantissa_len+2 个 token：
        符号、mantissa 分块、指数。这些 token 的取值空间有限且已知，
        因此可以一次性建好 numpy 查表，把逐 token 的字典查询换成数组索引。
        """
        fe = self.env.float_encoder
        w2id = self.env.float_word2id
        self._fe_precision = fe.float_precision
        self._fe_mantissa_len = fe.mantissa_len
        self._fe_base = fe.base
        self._fe_max_exponent = fe.max_exponent
        self._sign_ids = (w2id["+"], w2id["-"])
        self._mantissa_ids = np.array(
            [w2id["N" + f"%0{fe.base}d" % i] for i in range(fe.max_token)],
            dtype=np.int64,
        )
        self._exponent_ids = np.array(
            [w2id["E" + str(e)] for e in range(-fe.max_exponent, fe.max_exponent + 1)],
            dtype=np.int64,
        )
        # hint_encode 用：变量名 token 与 units_encode 结果的缓存
        self._var_name_ids = [
            w2id[f"x_{i}"] for i in range(self.params.max_input_dimension)
        ]
        self._y_name_id = w2id["y"]
        self._units_ids_cache = {}

    def compress(
        self, sequences_embeddings: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Takes: (N_max * (d_in+d_out)*(2+mantissa_len), B, d) tensors
        Returns: (N_max, B, d)
        """
        max_len, bs, float_descriptor_length, dim = sequences_embeddings.shape
        sequences_embeddings = sequences_embeddings.view(max_len, bs, -1)
        for layer in self.hidden_layers:
            sequences_embeddings = self.activation_fn(layer(sequences_embeddings))
        sequences_embeddings = self.fc(sequences_embeddings)
        return sequences_embeddings

    def forward(
        self, sequences, hints, packed: bool = False
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """`packed=True` 时 sequences 是 [(x_arr, y_arr), ...]，见 num_encode。"""
        sequences = self.num_encode(sequences, packed=packed)
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

    def _encode_values_to_ids(self, values: np.ndarray) -> np.ndarray:
        """把一维浮点数组编码成 shape (N, mantissa_len+2) 的 token id 数组。

        与 `FloatSequences.encode` 逐值编码等价：格式化仍走 printf，保证与原
        实现逐位一致，但字符串解析和字典查询全部换成 numpy 向量运算。
        """
        precision = self._fe_precision
        base = self._fe_base
        mantissa_len = self._fe_mantissa_len
        max_exponent = self._fe_max_exponent

        n = values.shape[0]
        out = np.empty((n, mantissa_len + 2), dtype=np.int64)
        if n == 0:
            return out
        if not np.isfinite(values).all():
            raise ValueError("num_encode 收到了 inf/nan，无法编码")

        # 取绝对值再格式化，等价于原实现的 m.lstrip("-")，
        # 同时让字符串宽度不受符号影响，便于按固定列位解析。
        fmt = "%." + str(precision) + "e"
        raw = np.array([fmt % v for v in np.abs(values)], dtype="S")
        raw = raw.view(np.uint8).reshape(n, -1)

        # 布局固定为 "d.ddd…e±XX"：第 0 位与第 2..precision+1 位是有效数字
        digits = np.empty((n, precision + 1), dtype=np.int64)
        digits[:, 0] = raw[:, 0].astype(np.int64) - 48
        digits[:, 1:] = raw[:, 2 : precision + 2].astype(np.int64) - 48

        # 指数：符号位紧跟 'e'，之后是 2~3 位数字（短的那些行尾部补 0 字节）
        sign_col = precision + 3
        exp_cols = raw[:, sign_col + 1 :]
        exponent = np.zeros(n, dtype=np.int64)
        for c in range(exp_cols.shape[1]):
            col = exp_cols[:, c]
            exponent = np.where(
                col >= 48, exponent * 10 + (col.astype(np.int64) - 48), exponent
            )
        exponent = np.where(raw[:, sign_col] == ord("-"), -exponent, exponent)
        exponent -= precision

        # 与原实现一致的下溢处理：mantissa 归零、指数归零
        underflow = exponent < -max_exponent
        if underflow.any():
            digits[underflow] = 0
            exponent[underflow] = 0
        overflow = exponent > max_exponent
        if overflow.any():
            raise ValueError(
                f"指数 {int(exponent[overflow][0])} 超出词表范围 "
                f"(max_exponent={max_exponent})"
            )

        out[:, 0] = np.where(values >= 0, self._sign_ids[0], self._sign_ids[1])
        for k in range(mantissa_len):
            chunk = digits[:, k * base : (k + 1) * base]
            packed = np.zeros(n, dtype=np.int64)
            for d in range(chunk.shape[1]):
                packed = packed * 10 + chunk[:, d]
            out[:, 1 + k] = self._mantissa_ids[packed]
        out[:, mantissa_len + 1] = self._exponent_ids[exponent + max_exponent]
        return out

    def num_encode(self, sequences, packed: bool = False) -> List[paddle.Tensor]:
        """向量化的数值编码。

        原实现逐 sequence、逐 point、逐 token 走 Python 循环 + 字典查询，
        profiler 实测占单步 CPU 时间的 32%。这里改成：先把整个 batch 的 x/y
        值拼成一条数组做一次编码，再用 numpy 切片拼装每条 sequence 的 token
        矩阵。输出与改造前的逐 point 实现逐元素一致。

        `packed=False`（默认，推理路径在用）时 sequences 是逐 point 的
        [(x, y), ...] 列表；`packed=True` 时是 [(x_arr, y_arr), ...]，其中
        x_arr 为 (n_points, n_vars)、y_arr 为 (n_points, n_out) 的整块数组。
        调用侧本来就持有整块数组，拆成逐 point 再在这里拼回去纯属浪费
        （实测占单步约 23%），所以训练路径走 packed。
        """
        dlen = self.float_scalar_descriptor_len
        max_in = self.params.max_input_dimension
        max_out = self.params.max_output_dimension
        row_len = (max_in + max_out) * dlen + 2

        # 第一遍：整理每条 sequence 的 x/y 数组，并把所有数值平铺到一起
        per_seq = []
        flat = []
        for seq in sequences:
            if packed:
                x_batch, y_batch = seq
                x_batch = np.asarray(x_batch, dtype=np.float64)
                y_batch = np.asarray(y_batch, dtype=np.float64)
            elif len(seq) == 0:
                per_seq.append(None)
                continue
            else:
                x_batch = np.asarray([x for x, _ in seq], dtype=np.float64)
                y_batch = np.asarray([y for _, y in seq], dtype=np.float64)
            if x_batch.ndim == 1:
                x_batch = x_batch.reshape(-1, 1)
            if y_batch.ndim == 1:
                y_batch = y_batch.reshape(-1, 1)
            if x_batch.shape[0] == 0:
                per_seq.append(None)
                continue
            if x_batch.shape[1] > max_in:
                raise ValueError(
                    f"输入维度 {x_batch.shape[1]} 超过最大允许维度 {max_in}"
                )
            per_seq.append((x_batch, y_batch))
            flat.append(x_batch.ravel())
            flat.append(y_batch.ravel())

        if not flat:
            return [paddle.to_tensor([], dtype="int64") for _ in sequences]

        # 第二遍：一次性编码整个 batch 的数值
        all_ids = self._encode_values_to_ids(np.concatenate(flat))

        # 第三遍：按 sequence 切回来，拼装 token id 矩阵
        res = []
        offset = 0
        for item in per_seq:
            if item is None:
                res.append(paddle.to_tensor([], dtype="int64"))
                continue
            x_batch, y_batch = item
            n_points, n_vars = x_batch.shape
            n_out = y_batch.shape[1]

            x_ids = all_ids[offset : offset + x_batch.size]
            offset += x_batch.size
            y_ids = all_ids[offset : offset + y_batch.size]
            offset += y_batch.size

            toks = np.empty((n_points, row_len), dtype=np.int64)
            toks[:, 0] = self.common_token_ids["<DATA_POINT>"]
            x_end = 1 + n_vars * dlen
            toks[:, 1:x_end] = x_ids.reshape(n_points, n_vars * dlen)
            in_end = 1 + max_in * dlen
            toks[:, x_end:in_end] = self.common_token_ids["<INPUT_PAD>"]
            y_end = in_end + n_out * dlen
            toks[:, in_end:y_end] = y_ids.reshape(n_points, n_out * dlen)
            toks[:, y_end : row_len - 1] = self.common_token_ids["<OUTPUT_PAD>"]
            toks[:, row_len - 1] = self.common_token_ids["</DATA_POINT>"]
            res.append(paddle.to_tensor(toks, dtype="int64"))
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

    def _units_to_ids(self, unit):
        """units_encode + 字典查询的缓存版本。

        units_encode 的结果只取决于那几个整数分量，同一 batch 内重复率很高，
        缓存掉可以省掉字符串拼接和逐 token 的字典查询。
        """
        key = unit if isinstance(unit, str) else tuple(np.asarray(unit).ravel().tolist())
        ids = self._units_ids_cache.get(key)
        if ids is None:
            w2id = self.env.float_word2id
            ids = [w2id[u] for u in self.env.equation_encoder.units_encode(unit)]
            self._units_ids_cache[key] = ids
        return ids

    def hint_encode(self, hints, use_hints):
        """向量化的提示编码。

        每行 hint 的长度固定为 float_vector_descriptor_len + 2（与 num_encode 的
        row_len 相同），且 pad 一律在尾部；所以先用 pad 填满 numpy 矩阵、再把
        前缀 token 写进去，等价于原实现而省掉逐行 extend([pad] * n) 的列表构建。
        const 里的浮点数在整个 batch 上一次性编码。
        输出与改造前的逐 token 实现逐元素一致。
        """
        use_hints_list = use_hints.split(",")
        row_len = self.float_vector_descriptor_len + 2
        hint_pad_id = self.env.float_word2id["<HINT_PAD>"]
        w2id = self.env.float_word2id
        n_seq = len(hints[0])

        # 复现原实现 current_hints_idx 的递增顺序，保持与 hints 列表的位置对应关系
        idx_of = {}
        cursor = 0
        for name in (
            "units",
            "complexity",
            "unarys",
            "add_structure",
            "mul_structure",
            "consts",
        ):
            if name in use_hints_list:
                idx_of[name] = cursor
                cursor += 1

        # const 的浮点值在整个 batch 上一次编码，再按遍历顺序取用
        const_ids = None
        const_cursor = 0
        if "consts" in idx_of:
            consts_all = hints[idx_of["consts"]]
            numeric = [
                _value
                for seq_id in range(n_seq)
                for _value, _ in consts_all[seq_id]
                if _value != "pi"
            ]
            if numeric:
                const_ids = self._encode_values_to_ids(
                    np.asarray(numeric, dtype=np.float64)
                )

        res = []
        for seq_id in range(n_seq):
            rows = []

            if "units" in idx_of:
                units = hints[idx_of["units"]][seq_id]
                last = len(units) - 1
                for i, unit in enumerate(units):
                    rows.append(
                        [
                            self.common_token_ids["<PHYSICAL_UNITS>"],
                            self._var_name_ids[i] if i != last else self._y_name_id,
                            *self._units_to_ids(unit),
                            self.common_token_ids["</PHYSICAL_UNITS>"],
                        ]
                    )

            if "complexity" in idx_of:
                for c in hints[idx_of["complexity"]][seq_id]:
                    # 复杂度可以是字符串(simple/middle/hard)或数字
                    if isinstance(c, str) or c != 0:
                        com_tok = f"COMPLEXITY:{c}"
                    else:
                        com_tok = "<UNKNOWN_COMPLEXITY>"
                    rows.append(
                        [
                            self.common_token_ids["<COMPLEXITY>"],
                            w2id[com_tok],
                            self.common_token_ids["</COMPLEXITY>"],
                        ]
                    )

            if "unarys" in idx_of:
                unarys = hints[idx_of["unarys"]][seq_id]
                rows.append(
                    [
                        self.common_token_ids["<UNARY>"],
                        *[w2id[u] for u in unarys],
                        self.common_token_ids["</UNARY>"],
                    ]
                )

            if "add_structure" in idx_of:
                for a in hints[idx_of["add_structure"]][seq_id]:
                    rows.append(
                        [
                            self.common_token_ids["<ADD_STRUCTURE>"],
                            *[self._var_name_ids[j] for j in a],
                            self.common_token_ids["</ADD_STRUCTURE>"],
                        ]
                    )

            if "mul_structure" in idx_of:
                for m in hints[idx_of["mul_structure"]][seq_id]:
                    rows.append(
                        [
                            self.common_token_ids["<MUL_STRUCTURE>"],
                            *[self._var_name_ids[j] for j in m],
                            self.common_token_ids["</MUL_STRUCTURE>"],
                        ]
                    )

            if "consts" in idx_of:
                for _value, _units in hints[idx_of["consts"]][seq_id]:
                    if _value != "pi":
                        value_ids = const_ids[const_cursor].tolist()
                        const_cursor += 1
                    else:
                        value_ids = [w2id["pi"]]
                    rows.append(
                        [
                            self.common_token_ids["<USED_CONST>"],
                            *value_ids,
                            *self._units_to_ids(_units),
                            self.common_token_ids["</USED_CONST>"],
                        ]
                    )

            arr = np.full((len(rows), row_len), hint_pad_id, dtype=np.int64)
            for r, row in enumerate(rows):
                arr[r, : len(row)] = row
            res.append(paddle.to_tensor(arr, dtype="int64"))
        return res
