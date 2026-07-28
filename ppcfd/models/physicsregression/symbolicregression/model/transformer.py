import itertools
import math
from logging import getLogger

import numpy as np
import paddle

from ...paddle_utils import *

N_MAX_POSITIONS = 4096
logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = paddle.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    paddle.nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        paddle.nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.detach_()
    out.stop_gradient = not False
    out[:, 0::2] = paddle.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = paddle.FloatTensor(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    if __debug__:  # 只在调试模式下检查，避免频繁GPU-CPU同步
        assert paddle.max(lengths).item() <= slen
    bs = lengths.shape[0]
    alen = paddle.arange(slen, dtype=paddle.long, device=lengths.device)
    mask = alen < lengths[:, None]
    if causal:
        attn_mask = paddle.tile(alen[None, None, :], (bs, slen, 1)) <= alen[None, :, None]
    else:
        attn_mask = mask
    assert tuple(mask.shape) == (bs, slen)
    assert causal is False or tuple(attn_mask.shape) == (bs, slen, slen)
    return mask, attn_mask


class MultiHeadAttention(paddle.nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, src_dim, dropout, normalized_attention):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.normalized_attention = normalized_attention
        assert self.dim % self.n_heads == 0
        self.q_lin = paddle.compat.nn.Linear(dim, dim)
        self.k_lin = paddle.compat.nn.Linear(src_dim, dim)
        self.v_lin = paddle.compat.nn.Linear(src_dim, dim)
        self.out_lin = paddle.compat.nn.Linear(dim, dim)
        if self.normalized_attention:
            self.attention_scale = paddle.nn.Parameter(
                paddle.tensor(1.0 / math.sqrt(dim // n_heads))
            )

    def forward(self, input, mask=None, kv=None, use_cache=False):
        """
        Self-attention (if kv is None)
        or attention over source sentence (provided by kv).
        Input is (bs, qlen, dim)
        Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.shape
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.shape[1]
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        n_heads = self.n_heads
        dim_per_head = dim // n_heads

        def shape(x):
            """projection"""
            return x.reshape(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """compute context"""
            return x.transpose(1, 2).reshape(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))
        if kv is None:
            k = shape(self.k_lin(input))
            v = shape(self.v_lin(input))
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))
            v = shape(self.v_lin(v))
        if use_cache:
            if self.layer_id in self.cache:
                if kv is None:
                    k_, v_ = self.cache[self.layer_id]
                    k = paddle.cat([k_, k], dim=2)
                    v = paddle.cat([v_, v], dim=2)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = k, v
        if self.normalized_attention:
            q = paddle.nn.functional.normalize(q, p=2, dim=-1)
            k = paddle.nn.functional.normalize(k, p=2, dim=-1)
            q = q * self.attention_scale
        else:
            q = q / math.sqrt(dim_per_head)
        scores = paddle.matmul(q, k.transpose(2, 3))
        if mask is not None:
            mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)
            mask = (mask == 0).reshape(mask_reshape).expand_as(scores)
            scores.masked_fill_(mask, -float("inf"))
        weights = paddle.compat.nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )
        weights = paddle.nn.functional.dropout(
            weights, p=self.dropout, training=self.training
        )
        context = paddle.matmul(weights, v)
        context = unshape(context)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = weights.detach().cpu()
        return self.out_lin(context)


class TransformerFFN(paddle.nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, hidden_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.midlin = paddle.nn.ModuleList()
        self.lin1 = paddle.compat.nn.Linear(in_dim, dim_hidden)
        for i in range(1, self.hidden_layers):
            self.midlin.append(paddle.compat.nn.Linear(dim_hidden, dim_hidden))
        self.lin2 = paddle.compat.nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = paddle.nn.functional.relu(x=x)
        for mlin in self.midlin:
            x = mlin(x)
            x = paddle.nn.functional.relu(x=x)
        x = self.lin2(x)
        x = paddle.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerModel(paddle.nn.Module):
    STORE_OUTPUTS = True

    def __init__(
        self,
        params,
        id2word,
        is_encoder,
        with_output,
        use_prior_embeddings,
        positional_embeddings,
    ):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()
        self.dtype = paddle.float16 if params.fp16 else paddle.float32
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.sep_index = self.word2id["<SEP>"]
        self.n_words = len(self.id2word)
        assert len(self.id2word) == self.n_words
        if params.use_dimension_mask:
            self.use_dimension_mask = True
            self.variables_start_index = self.word2id["x_0"]
            self.variables_maxi_num = max(
                [
                    (
                        i
                        if self.id2word[self.variables_start_index + i - 1]
                        == f"x_{i - 1}"
                        else 0
                    )
                    for i in range(1, 100)
                ]
            )
            self.dimensionless_op_index = paddle.tensor(
                [
                    self.word2id[op]
                    for op in [
                        "arccos",
                        "arcsin",
                        "arctan",
                        "cos",
                        "cosh",
                        "exp",
                        "log",
                        "pi",
                        "sin",
                        "sinh",
                        "tan",
                    ]
                ]
            )
            self.dimensionless_dimension_index = paddle.tensor(
                [self.word2id[dim] for dim in ["M0", "S0", "K0", "T0", "V0"]]
            )
            self.dimension_start_index = min(
                [self.word2id.get(f"M-{i}", 10000000000.0) for i in range(101)]
            )
            self.dimension_end_index = max(
                [self.word2id.get(f"V{i}", -10000000000.0) for i in range(101)]
            )
            dimension0 = [self.word2id[f"{d}0"] for d in ["M", "S", "K", "T", "V"]]
            self.dimension_num = [
                dimension0[0] - self.dimension_start_index + 1,
                dimension0[1] - dimension0[0],
                dimension0[2] - dimension0[1],
                dimension0[3] - dimension0[2],
                dimension0[4] - dimension0[3],
            ]
            self.dimension_num = [dimension0[0] - self.dimension_start_index]
            for i in range(1, 5):
                self.dimension_num.append(
                    dimension0[i] - dimension0[i - 1] - self.dimension_num[-1] - 1
                )
            assert self.dimension_num[-1] == self.dimension_end_index - dimension0[-1]
            self.dimension_mask = paddle.tensor(
                [
                    [
                        (
                            True
                            if dimension0[i] - self.dimension_num[i]
                            <= j
                            <= dimension0[i] + self.dimension_num[i]
                            else False
                        )
                        for j in range(self.n_words)
                    ]
                    for i in range(5)
                ]
            )
        else:
            self.use_dimension_mask = False
        self.dim = params.enc_emb_dim if is_encoder else params.dec_emb_dim
        self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4
        self.n_hidden_layers = (
            params.n_enc_hidden_layers if is_encoder else params.n_dec_hidden_layers
        )
        self.n_heads = params.n_enc_heads if is_encoder else params.n_dec_heads
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        self.norm_attention = params.norm_attention
        assert (
            self.dim % self.n_heads == 0
        ), "transformer dim must be a multiple of n_heads"
        if positional_embeddings is None or positional_embeddings == "alibi":
            self.position_embeddings = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
            )
        elif positional_embeddings == "learnable":
            self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        else:
            raise NotImplementedError
        self.use_prior_embeddings = use_prior_embeddings
        if not use_prior_embeddings:
            self.embeddings = Embedding(
                self.n_words, self.dim, padding_idx=self.pad_index
            )
        else:
            self.embeddings = None
        self.layer_norm_emb = paddle.nn.LayerNorm(self.dim, eps=1e-12)
        self.attentions = paddle.nn.ModuleList()
        self.layer_norm1 = paddle.nn.ModuleList()
        self.ffns = paddle.nn.ModuleList()
        self.layer_norm2 = paddle.nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = paddle.nn.ModuleList()
            self.encoder_attn = paddle.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    self.dim,
                    dropout=self.attention_dropout,
                    normalized_attention=self.norm_attention,
                )
            )
            self.layer_norm1.append(paddle.nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(paddle.nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        self.src_dim,
                        dropout=self.attention_dropout,
                        normalized_attention=self.norm_attention,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    self.n_hidden_layers,
                    dropout=self.dropout,
                )
            )
            self.layer_norm2.append(paddle.nn.LayerNorm(self.dim, eps=1e-12))
        self.cache = None
        if self.with_output:
            assert not self.use_prior_embeddings
            self.proj = paddle.compat.nn.Linear(self.dim, self.n_words, bias=True)
            if params.share_inout_emb and False:
                self.proj.weight = self.embeddings.weight
        if self.is_decoder and params.decode_physical_units == "double-seq":
            self.units_enc = paddle.compat.nn.Linear(self.dim * 5, self.dim)
            self.units_dec = paddle.compat.nn.Linear(self.dim, self.dim * 5)

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        x,
        lengths,
        causal,
        src_enc=None,
        src_len=None,
        positions=None,
        use_cache=False,
        units=None,
    ):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `dim` LongTensor(slen, bs, 5), containing dimensions
        """
        slen, bs = x.shape[:2]
        assert lengths.shape[0] == bs
        if __debug__:  # 只在调试模式下检查，避免频繁GPU-CPU同步
            assert paddle.max(lengths).item() <= slen
        x = x.transpose(0, 1)
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.shape[0] == bs
        assert not (use_cache and self.cache is None)
        if self.is_decoder and units is not None:
            units = units.transpose(0, 1)
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = (
                paddle.arange(paddle.max(src_len), dtype=paddle.long, device=lengths.device)
                < src_len[:, None]
            )
        if positions is None:
            # PaddlePaddle: 直接使用 paddle.arange 创建位置张量
            positions = paddle.arange(slen, dtype='int64').unsqueeze(0)
        else:
            assert tuple(positions.shape) == (slen, bs)
            positions = positions.transpose(0, 1)
        if use_cache:
            _slen = slen - self.cache["slen"]
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
            if self.is_decoder and units is not None:
                units = units[:, -_slen:]
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []
        if not self.use_prior_embeddings:
            tensor = self.embeddings(x)
        else:
            tensor = x
        if self.is_decoder and units is not None:
            units_tensor = self.embeddings(units)
            units_tensor = units_tensor.reshape(
                (units_tensor.shape[0], units_tensor.shape[1], -1)
            )
            units_tensor = self.units_enc(units_tensor)
            tensor = tensor + units_tensor
        if self.position_embeddings is not None:
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.layer_norm_emb(tensor)
        tensor = paddle.nn.functional.dropout(
            tensor, p=self.dropout, training=self.training
        )
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())
        for i in range(self.n_layers):
            self.attentions[i].cache = self.cache
            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)
            attn = paddle.nn.functional.dropout(
                attn, p=self.dropout, training=self.training
            )
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
            if self.is_decoder and src_enc is not None:
                self.encoder_attn[i].cache = self.cache
                attn = self.encoder_attn[i](
                    tensor, src_mask, kv=src_enc, use_cache=use_cache
                )
                attn = paddle.nn.functional.dropout(
                    attn, p=self.dropout, training=self.training
                )
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())
        if use_cache:
            self.cache["slen"] += tensor.shape[1]
        tensor = tensor.transpose(0, 1)
        return tensor

    def predict(self, tensor, pred_mask, y, y_units=None, get_scores=False):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].reshape(-1, self.dim)
        if __debug__:  # 只在调试模式下检查
            assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x)
        loss = paddle.nn.functional.cross_entropy(
            input=scores.float(), label=y, reduction="mean"
        )
        next_word = paddle.topk(scores, 1)[1].squeeze(1)
        if y_units is not None:
            x_dim = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].reshape(-1, self.dim)
            if __debug__:  # 只在调试模式下检查
                assert (y_units == self.pad_index).sum().item() == 0
            latent_units = self.units_dec(x_dim).reshape(-1, self.dim)
            scores_units = self.proj(latent_units)
            loss_units = paddle.nn.functional.cross_entropy(
                input=scores_units.float(), label=y_units, reduction="mean"
            )
            next_unit = paddle.topk(scores_units, 1)[1].squeeze(1)
            loss = loss + loss_units * 0.2
        return scores, loss

    def generate(
        self,
        src_enc,
        src_len,
        decode_physical_units,
        max_len=200,
        top_p=1.0,
        sample_temperature=None,
        units=None,
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """
        if decode_physical_units == "single-seq":
            max_len *= 5
        if decode_physical_units is None or decode_physical_units == "single-seq":
            bs = len(src_len)
            assert src_enc.shape[0] == bs
            # PaddlePaddle: 使用 paddle.full 创建填充张量
            generated = paddle.full([max_len, bs], self.pad_index, dtype=src_len.dtype)
            generated[0].fill_(self.eos_index)
            # PaddlePaddle: 直接创建并扩展位置张量
            positions = paddle.arange(max_len, dtype='int64').unsqueeze(1).expand([max_len, bs])
            cur_len = 1
            gen_len = src_len.clone().fill_(1)
            unfinished_sents = src_len.clone().fill_(1)
            word_perplexity = paddle.zeros(bs).to(src_enc.device)
            self.cache = {"slen": 0}
            while cur_len < max_len:
                tensor = self.forward(
                    "fwd",
                    x=generated[:cur_len],
                    lengths=gen_len,
                    positions=positions[:cur_len],
                    causal=True,
                    src_enc=src_enc,
                    src_len=src_len,
                    use_cache=True,
                )
                assert tuple(tensor.shape) == (1, bs, self.dim)
                tensor = tensor.data[-1, :, :].to(self.dtype)
                scores = self.proj(tensor)
                if sample_temperature is None:
                    next_words = paddle.topk(scores, 1)[1].squeeze(1)
                else:
                    next_words = paddle.multinomial(
                        paddle.compat.nn.functional.softmax(
                            scores.float() / sample_temperature, dim=1
                        ),
                        num_samples=1,
                    ).squeeze(1)
                assert tuple(next_words.shape) == (bs,)
                generated[cur_len] = next_words * unfinished_sents + self.pad_index * (
                    1 - unfinished_sents
                )
                next_words_prob = paddle.compat.nn.functional.softmax(
                    scores.float(), dim=1
                )
                next_words_perplexity = next_words_prob[
                    tuple([paddle.arange(bs), next_words])
                ]
                assert tuple(next_words_perplexity.shape) == (bs,)
                word_perplexity.add_(
                    # PaddlePaddle: 显式类型转换 float32 * int64 -> float32
                    paddle.log(next_words_perplexity.detach()) * unfinished_sents.astype('float32')
                )
                gen_len.add_(unfinished_sents)
                # PaddlePaddle: .ne() 需要tensor参数，改用 != 运算符
                unfinished_sents.mul_((next_words != self.eos_index).astype('int64'))
                cur_len = cur_len + 1
                if paddle.max(unfinished_sents) == 0:
                    break
            if cur_len == max_len:
                generated[-1].masked_fill_(unfinished_sents.bool(), self.eos_index)
            assert (generated == self.eos_index).sum() == 2 * bs
            generated = generated.unsqueeze(-1).view(generated.shape[0], bs)
            rows, cols = paddle.nonzero(generated[1:] == self.eos_index, as_tuple=True)
            # PaddlePaddle: 显式转换 int64 -> float32
            word_perplexity = paddle.exp(word_perplexity / rows.astype('float32'))
            return generated[:cur_len], gen_len, None, word_perplexity, None
        elif decode_physical_units == "double-seq":
            bs = len(src_len)
            assert src_enc.shape[0] == bs
            # PaddlePaddle: 使用 paddle.full 创建填充张量
            generated1 = paddle.full([max_len, bs], self.pad_index, dtype=src_len.dtype)
            generated1[0].fill_(self.eos_index)
            generated2 = paddle.full([max_len, bs, 5], self.pad_index, dtype=src_len.dtype)
            generated2[0].fill_(self.eos_index)
            # PaddlePaddle: 直接创建并扩展位置张量
            positions = paddle.arange(max_len, dtype='int64').unsqueeze(1).expand([max_len, bs])
            cur_len = 1
            gen_len = src_len.clone().fill_(1)
            unfinished_sents = src_len.clone().fill_(1)
            word_perplexity = paddle.zeros(bs).to(src_enc.device)
            dimension_perplexity = paddle.zeros(bs).to(src_enc.device)
            self.cache = {"slen": 0}
            while cur_len < max_len:
                tensor = self.forward(
                    "fwd",
                    x=generated1[:cur_len],
                    lengths=gen_len,
                    positions=positions[:cur_len],
                    causal=True,
                    src_enc=src_enc,
                    src_len=src_len,
                    use_cache=True,
                    units=generated2[:cur_len],
                )
                assert tuple(tensor.shape) == (1, bs, self.dim)
                tensor = tensor.data[-1, :, :].to(self.dtype)
                scores = self.proj(tensor)
                if self.use_dimension_mask:
                    dimension_mask = paddle.ones((bs, self.n_words))
                    required_dimension = generated2[cur_len - 1].clone().to("cpu")
                    variable_allowed = [
                        paddle.all(units[i] == required_dimension[i], dim=1)
                        for i in range(bs)
                    ]
                    for i in range(bs):
                        dimension_mask[i][
                            self.variables_start_index : self.variables_start_index
                            + len(variable_allowed[i])
                        ][~variable_allowed[i]] = 0
                        dimension_mask[i][
                            self.variables_start_index
                            + len(variable_allowed[i]) : self.variables_start_index
                            + self.variables_maxi_num
                        ] = 0
                    dimensionless_allowed = paddle.all(
                        required_dimension == self.dimensionless_dimension_index, dim=1
                    )
                    if any(~dimensionless_allowed):
                        mask_idx = paddle.meshgrid(
                            paddle.arange(bs)[~dimensionless_allowed],
                            self.dimensionless_op_index,
                            indexing="ij",
                        )
                        dimension_mask[mask_idx] = 0
                    masked_scores = scores.clone()
                    masked_scores[dimension_mask == 0] = -paddle.inf
                else:
                    masked_scores = scores
                if sample_temperature is None:
                    next_words = paddle.topk(masked_scores, 1)[1].squeeze(1)
                else:
                    next_words = paddle.multinomial(
                        paddle.compat.nn.functional.softmax(
                            masked_scores.float() / sample_temperature, dim=1
                        ),
                        num_samples=1,
                    ).squeeze(1)
                assert tuple(next_words.shape) == (bs,)
                generated1[cur_len] = next_words * unfinished_sents + self.pad_index * (
                    1 - unfinished_sents
                )
                latent_units = self.units_dec(tensor).view(-1, 5, self.dim)
                scores_units = self.proj(latent_units).view(-1, 5, self.n_words)
                if self.use_dimension_mask:
                    masked_scores_units = scores_units.clone()
                    masked_scores_units[
                        ~self.dimension_mask.unsqueeze(0).expand_as(scores_units)
                    ] = -paddle.inf
                else:
                    masked_scores_units = scores_units
                if sample_temperature is None:
                    next_words_dim = paddle.topk(masked_scores_units, 1)[1].squeeze(-1)
                else:
                    next_words_dim = (
                        paddle.multinomial(
                            paddle.compat.nn.functional.softmax(
                                masked_scores_units.reshape(-1, self.n_words).float()
                                / sample_temperature,
                                dim=-1,
                            ),
                            num_samples=1,
                        )
                        .squeeze(-1)
                        .reshape(bs, 5)
                    )
                assert tuple(next_words_dim.shape) == (bs, 5)
                generated2[cur_len] = next_words_dim * unfinished_sents[
                    :, None
                ] + self.pad_index * (1 - unfinished_sents[:, None])
                next_words_prob = paddle.compat.nn.functional.softmax(
                    scores.float(), dim=1
                )
                next_words_perplexity = next_words_prob[
                    tuple([paddle.arange(bs), next_words])
                ]
                next_dimensions_prob = paddle.compat.nn.functional.softmax(
                    scores_units.float(), dim=-1
                )
                next_dimensions_perplexity = next_dimensions_prob[
                    tuple(
                        [
                            paddle.arange(bs).unsqueeze(1).repeat((1, 5)).reshape(-1),
                            paddle.arange(5).unsqueeze(0).repeat((bs, 1)).reshape(-1),
                            next_words_dim.reshape(-1),
                        ]
                    )
                ].reshape((bs, 5))
                next_dimensions_perplexity = paddle.prod(
                    next_dimensions_perplexity, dim=1
                )
                assert (
                    tuple(next_words_perplexity.shape)
                    == tuple(next_dimensions_perplexity.shape)
                    == (bs,)
                )
                word_perplexity.add_(
                    # PaddlePaddle: 显式类型转换 float32 * int64 -> float32
                    paddle.log(next_words_perplexity.detach()) * unfinished_sents.astype('float32')
                )
                dimension_perplexity.add_(
                    # PaddlePaddle: 显式类型转换 float32 * int64 -> float32
                    paddle.log(next_dimensions_perplexity.detach()) * unfinished_sents.astype('float32')
                )
                gen_len.add_(unfinished_sents)
                # PaddlePaddle: .ne() 需要tensor参数，改用 != 运算符
                unfinished_sents.mul_((next_words != self.eos_index).astype('int64'))
                cur_len = cur_len + 1
                if paddle.max(unfinished_sents) == 0:
                    break
            if cur_len == max_len:
                generated1[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
            assert (generated1 == self.eos_index).sum() == 2 * bs
            generated1 = generated1.unsqueeze(-1).view(generated1.shape[0], bs)
            rows, cols = paddle.nonzero(generated1[1:] == self.eos_index, as_tuple=True)
            # PaddlePaddle: 显式转换 int64 -> float32
            word_perplexity = paddle.exp(-word_perplexity / rows.astype('float32'))
            dimension_perplexity = paddle.exp(-dimension_perplexity / rows.astype('float32') / 5)
            return (
                generated1[:cur_len],
                gen_len,
                generated2[:cur_len],
                word_perplexity,
                dimension_perplexity,
            )
        else:
            raise ValueError()

    def generate_beam(
        self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """
        assert src_enc.shape[0] == src_len.shape[0]
        assert beam_size >= 1
        bs = len(src_len)
        n_words = self.n_words
        src_enc = (
            src_enc.unsqueeze(1)
            .expand((bs, beam_size) + src_enc.shape[1:])
            .contiguous()
            .view((bs * beam_size,) + src_enc.shape[1:])
        )
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)
        # PaddlePaddle: 使用 paddle.full 创建填充张量
        generated = paddle.full([max_len, bs * beam_size], self.pad_index, dtype=src_len.dtype)
        generated[0].fill_(self.eos_index)
        generated_hyps = [
            BeamHypotheses(beam_size, max_len, length_penalty, early_stopping)
            for _ in range(bs)
        ]
        # PaddlePaddle: 直接创建并扩展位置张量
        positions = paddle.arange(max_len, dtype='int64').unsqueeze(1).expand_as(generated)
        # PaddlePaddle: 使用 paddle.full 创建 beam_scores
        beam_scores = paddle.full([bs, beam_size], 0.0, dtype='float32')
        beam_scores[:, 1:] = -1000000000.0
        beam_scores = beam_scores.view(-1)
        cur_len = 1
        self.cache = {"slen": 0}
        done = [(False) for _ in range(bs)]
        while cur_len < max_len:
            tensor = self.forward(
                "fwd",
                x=generated[:cur_len],
                lengths=paddle.full([bs * beam_size], cur_len, dtype=src_len.dtype),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=True,
            )
            assert tuple(tensor.shape) == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]
            scores = self.proj(tensor)
            scores = paddle.nn.functional.log_softmax(x=scores.float(), axis=-1)
            assert tuple(scores.shape) == (bs * beam_size, n_words)
            _scores = scores + beam_scores[:, None].expand_as(scores)
            _scores = _scores.view(bs, beam_size * n_words)
            next_scores, next_words = paddle.topk(
                _scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            assert tuple(next_scores.shape) == tuple(next_words.shape) == (bs, 2 * beam_size)
            next_batch_beam = []
            for sent_id in range(bs):
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    paddle.max(next_scores[sent_id]).item()
                )
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)
                    continue
                next_sent_beam = []
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    beam_id = paddle.div(idx, n_words, rounding_mode="trunc")
                    word_id = idx % n_words
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(
                            generated[:cur_len, sent_id * beam_size + beam_id]
                            .clone()
                            .cpu(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append(
                            (value, word_id, sent_id * beam_size + beam_id)
                        )
                    if len(next_sent_beam) == beam_size:
                        break
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)
            assert len(next_batch_beam) == bs * beam_size
            # PaddlePaddle: 使用 paddle.to_tensor 从列表创建张量
            beam_scores = paddle.to_tensor([x[0] for x in next_batch_beam], dtype='float32')
            beam_words = paddle.to_tensor([x[1] for x in next_batch_beam], dtype=generated.dtype)
            beam_idx = paddle.to_tensor([x[2] for x in next_batch_beam], dtype=src_len.dtype)
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            cur_len = cur_len + 1
            if all(done):
                break
        # PaddlePaddle: 创建目标长度张量
        tgt_len = paddle.zeros([bs], dtype=src_len.dtype)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1
            best.append(best_hyp)
        # PaddlePaddle: 使用 paddle.full 创建解码张量
        decoded = paddle.full([int(paddle.max(tgt_len).item()), bs], self.pad_index, dtype=src_len.dtype)
        for i, hypo in enumerate(best):
            decoded[: tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index
        assert (decoded == self.eos_index).sum() == 2 * bs
        return decoded, tgt_len, generated_hyps


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.hyp)]
                )
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap,
        then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return (
                self.worst_score
                >= best_sum_logprobs / self.max_len**self.length_penalty
            )


def top_k_top_p_filtering(
    logits: paddle.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> paddle.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(
            top_k=top_k,
            filter_value=filter_value,
            min_tokens_to_keep=min_tokens_to_keep,
        )(None, logits)
    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )
    return logits


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(
        self, input_ids: paddle.LongTensor, scores: paddle.FloatTensor
    ) -> paddle.FloatTensor:
        """Torch method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TopKLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {top_k}"
            )
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: paddle.LongTensor, scores: paddle.FloatTensor
    ) -> paddle.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.shape[-1])
        indices_to_remove = scores < paddle.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_p: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(
        self, input_ids: paddle.LongTensor, scores: paddle.FloatTensor
    ) -> paddle.FloatTensor:
        sorted_logits, sorted_indices = paddle.compat.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
