"""
This file contains complete transformer encoder/decoder modules.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from .attention_utils import (
    Embedding,
    SinusoidalPE,
    LearnablePE,
    CausalTransformerDecoder,
    CausalTransformerDecoderLayer,
    OperatorDecoderLayer,
    get_block_attn_mask,
    get_embeddings,
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    GroupNorm,
    _generate_square_subsequent_mask,
)
from logging import getLogger
from functools import partial

logger = getLogger()


class PaddleRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=None, epsilon=1e-6, **kwargs):
        super().__init__()
        if eps is not None:
            epsilon = eps
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.epsilon = epsilon

    def forward(self, x):
        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        x = x * paddle.rsqrt(variance + self.epsilon)
        return x * self.weight


def resolve_norm(norm_name):
    if norm_name == "group":
        return partial(GroupNorm, 8)
    if norm_name == "rms":
        return getattr(nn, "RMSNorm", PaddleRMSNorm)
    return nn.LayerNorm


"""
Transformer Data modules

"""


class TransformerDataEncoder(nn.Module):
    """
    Encoder Transformer for data
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            if config.get("custom_encoder", 0):
                norm = resolve_norm(config.get("norm", "layer"))

                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=config.rotary,
                        custom_attn=config.custom_attn,
                        norm=norm,
                    ),
                    num_layers=config.n_layer,
                    norm=norm(config.dim_emb) if config.norm_first else None,
                    config=config,
                )
            else:
                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=False,
                        custom_attn=False,
                        norm=nn.LayerNorm,
                    ),
                    num_layers=config.n_layer,
                    norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
                    config=None,
                )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        """
        x: Tensor (bs, slen, dim)
        """

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


class TransformerDataDecoder(nn.Module):
    """
    Encoder-decoder Transformer for data (autoregressive)
    """

    def __init__(self, config, output_dim):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if self.config.kv_cache:
            self.transformer_decoder = CausalTransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    normalize_before=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )
        else:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    normalize_before=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        self.post_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.GELU(), nn.Linear(self.dim, output_dim))

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
    ):
        """
        Inputs:
            tgt:    Tensor (bs, output_len, dim)
                    should be data in the range [input_len-1, max_len-1)
            memory: Tensor (bs, input_len, dim)
                    output from encoder
        Output:
            tgt_output: Tensor (bs, output_len, output_dim)
                        should corresponds to data in the range [input_len, max_len)
        """

        if self.positional_embedding is not None:
            tgt = self.positional_embedding(tgt)  # (bs, output_len, dim)

        if self.config.kv_cache:
            tgt = tgt.transpose(0, 1)  # (output_len, bs, dim)
            memory = memory.transpose(0, 1)  # (input_len, bs, dim)
            tgt_mask = None  # (causal decoder handles this automatically)
            decoded = self.transformer_decoder(
                tgt=tgt, memory=memory, tgt_mask=tgt_mask
            )  # (output_len, bs, dim)
            tgt_output = self.post_proj(decoded).transpose(0, 1)  # (bs, output_len, output_dim)
        else:
            tgt_mask = _generate_square_subsequent_mask(tgt.size(1), dtype=tgt.dtype)
            decoded = self.transformer_decoder(
                tgt=tgt, memory=memory, tgt_mask=tgt_mask
            )  # (bs, output_len, dim)
            tgt_output = self.post_proj(decoded)  # (bs, output_len, output_dim)

        return tgt_output

    def generate(
        self,
        encoded,
        initial,
        input_times,
        output_times,
        all_times,
        pre_proj,
    ):
        """
        For evaluation/testing only.
        Inputs:
            encoded:      Tensor (input_len, bs, dim)
            initial:      Tensor (bs, query_dim+data_dim)
            input_times:  Tensor (input_len, query_dim)
            output_times: Tensor (output_len, query_dim)
            all_times:    Tensor (max_len, query_dim)
            pre_proj:     Projection for data input
        Output:
            data_output:  Tensor (output_len, bs, data_dim)

        """
        cur_len = 1
        output_len = output_times.size(0)
        bs = initial.size(0)
        query_dim = output_times.size(1)
        data_dim = initial.size(1) - query_dim
        generated = paddle.zeros([output_len, bs, data_dim], dtype=initial.dtype)

        cache = None
        tgt = pre_proj(initial)[None]  # (1, bs, dim)

        # generation loop
        while cur_len <= output_len:  # max length of generation

            if self.config.kv_cache:
                decoded, cache = self.transformer_decoder(tgt=tgt, memory=encoded, cache=cache)  # (cur_len, bs, dim)
                new_data = self.post_proj(decoded[-1])  # (bs, data_dim)
            else:
                tgt_batch = tgt.transpose([1, 0, 2])
                encoded_batch = encoded.transpose([1, 0, 2])
                tgt_mask = _generate_square_subsequent_mask(
                    tgt_batch.size(1), dtype=tgt_batch.dtype
                )
                decoded = self.transformer_decoder(
                    tgt=tgt_batch, memory=encoded_batch, tgt_mask=tgt_mask
                )  # (bs, cur_len, dim)
                new_data = self.post_proj(decoded[:, -1])  # (bs, data_dim)

            generated[cur_len - 1] = new_data

            new_input = paddle.cat(
                [output_times[cur_len - 1][None].expand(bs, query_dim), new_data], dim=1
            )  # (bs, query_dim + data_dim)

            tgt = paddle.cat([tgt, pre_proj(new_input[None])], axis=0)  # (cur_len + 1, bs, dim)

            cur_len += 1

        return generated


class DataOperatorDecoder(nn.Module):
    """
    Operator Decoder for data
    """

    def __init__(self, config, output_len=1, space_len=None):
        super().__init__()

        self.config = config

        self.dim = config.dim_emb

        self.time_embed_type = config.get("time_embed", "continuous")
        if self.time_embed_type == "continuous":
            self.time_proj = nn.Sequential(
                nn.Linear(config.query_dim, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.time_embed = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if space_len is None:
            space_len = config.patch_num_output**2

        self.patch_position_embeddings = get_embeddings((1, 1, space_len, self.dim))

        if config.self_attn > 0:
            # self attn + cross attn + ffn

            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    normalize_before=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

            if config.self_attn == 1:
                # self attn is restricted to patches with the same t
                self_attn_mask = get_block_attn_mask(
                    block_size=config.patch_num_output * config.patch_num_output, n_repeat=output_len
                )
                self.register_buffer("self_attn_mask", self_attn_mask)
        else:
            # cross attn + ffn

            norm = resolve_norm(config.get("norm", "layer"))

            self.transformer_decoder = nn.LayerList(
                [
                    OperatorDecoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        custom_attn=config.get("custom_attn", 0),
                        norm=norm,
                    )
                    for _ in range(config.n_layer)
                ]
            )
            self.decoder_norm = norm(config.dim_emb) if (config.norm_first and config.get("final_ln", True)) else None

    def get_query_emb(self, times):
        """
        Input:
            times:     Tensor (bs/1, output_len, 1)
        Output:
            query_emb: Tensor (bs/1, query_len, dim)
                       query_len = output_len * patch_num * patch_num
        """

        bs, output_len, query_dim = times.size()

        if self.time_embed_type == "continuous":
            times = self.time_proj(times)[:, :, None]  # (bs/1, output_len, 1, dim)
        else:
            times = self.time_embed[:, :output_len]  # (1, input_len, 1, dim)

        return (times + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    def forward(self, src, query_emb, src_key_padding_mask=None, tgt_mask=None):
        """
        src:         Tensor (bs, src_len, dim)
        query_emb:   Tensor (bs, query_len, dim)
        src_key_padding_mask: Optional[Tensor] (bs, src_len)
        tgt_mask:             Optional[Tensor] (query_len, query_len) or (bs*n_head, query_len, query_len)
        """

        if tgt_mask is None and self.config.self_attn == 1:
            tgt_mask = self.self_attn_mask

        if self.config.self_attn > 0:
            x = self.transformer_decoder(
                query_emb.transpose([1, 0, 2]),
                src.transpose([1, 0, 2]),
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            ).transpose([1, 0, 2])
        else:
            decoded = query_emb
            for layer in self.transformer_decoder:
                decoded = layer(
                    tgt=decoded,
                    memory=src,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
            if self.decoder_norm is not None:
                decoded = self.decoder_norm(decoded)
            x = decoded

        return x  # (bs, query_len, dim)


"""
Transformer Symbol Modules

"""


class TransformerSymbolEncoder(nn.Module):
    """
    Encoder Transformer for Symbols
    """

    def __init__(self, config, id2word):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            if config.get("custom_encoder", 0):
                norm = resolve_norm(config.get("norm", "layer"))

                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=config.rotary,
                        custom_attn=config.custom_attn,
                        norm=norm,
                    ),
                    num_layers=config.n_layer,
                    norm=norm(config.dim_emb) if config.norm_first else None,
                    config=config,
                )
            else:
                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=False,
                        custom_attn=False,
                        norm=nn.LayerNorm,
                    ),
                    num_layers=config.n_layer,
                    norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
                    config=None,
                )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)

        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

    def forward(self, x, mask=None, src_key_padding_mask=None, is_causal=None):
        """
        x:                    LongTensor (bs, slen)
        mask:                 Optional[Tensor] (bs, slen, slen)
        src_key_padding_mask: Optional[BoolTensor] (bs, slen)         (positions with value True will be ignored)
        """

        x = self.word_embeddings(x)  # (bs, slen, dim)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)  # (bs, slen, dim)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, mask, src_key_padding_mask, is_causal)

        return x  # (bs, slen, dim)


class TransformerSymbolDecoder(nn.Module):
    """
    Encoder-decoder Transformer for Symbols
    """

    def __init__(self, config, id2word):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if self.config.kv_cache:
            self.transformer_decoder = CausalTransformerDecoder(
                CausalTransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    normalize_before=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )
        else:
            self.transformer_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=config.dim_emb,
                    nhead=config.n_head,
                    dim_feedforward=config.dim_ffn,
                    dropout=config.dropout,
                    activation="gelu",
                    normalize_before=config.norm_first,
                ),
                num_layers=config.n_layer,
                norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
            )

        if config.positional_embedding is None:
            self.positional_embedding = None
        elif config.positional_embedding == "sinusoidal":
            self.positional_embedding = SinusoidalPE(config.dim_emb, config.dropout)
        elif config.positional_embedding == "learnable":
            self.positional_embedding = LearnablePE(config.dim_emb, config.dropout)
        else:
            raise NotImplementedError(f"Unknown positional embedding {config.positional_embedding}")

        # dictionary

        self.id2word = id2word
        self.word2id = {s: i for i, s in self.id2word.items()}
        self.bos_index = self.word2id["<BOS>"]
        self.eos_index = self.word2id["<EOS>"]
        self.pad_index = self.word2id["<PAD>"]
        self.n_words = len(self.id2word)

        self.word_embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)

        # output layer
        self.proj = nn.Linear(self.dim, self.n_words, bias=True)
        if config.share_inout_emb:
            self.proj.weight = self.word_embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        tgt,
        memory,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Inputs:
            tgt:                     LongTensor (bs, output_len)
            memory:                  LongTensor (bs, input_len, dim)        (output from encoder)
            tgt_key_padding_mask:    Optional[BoolTensor] (bs, output_len)  (True for positions that should be ignored)
            memory_key_padding_mask: Optional[BoolTensor] (bs, input_len)
        Output:
            decoded:                 Tensor (bs, output_len, dim)
        """

        tgt = self.word_embeddings(tgt)  # (bs, output_len, dim)

        if self.positional_embedding is not None:
            tgt = self.positional_embedding(tgt)  # (bs, output_len, dim)

        if self.config.kv_cache:
            tgt = tgt.transpose(0, 1)  # (output_len, bs, dim)
            memory = memory.transpose(0, 1)  # (input_len, bs, dim)
            tgt_mask = None  # (causal decoder handles this automatically)
            decoded = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
            )  # (output_len, bs, dim)
            return decoded.transpose(0, 1)  # (bs, output_len, dim)
        else:
            tgt_mask = _generate_square_subsequent_mask(tgt.size(1), dtype=tgt.dtype)
            decoded = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
            )  # (bs, output_len, dim)
            return decoded

    def predict(self, output, pred_mask, y):
        """
        Given the last hidden state, compute word scores and the loss.
        Inputs:
            output       Tensor     (bs, output_len, dim)
            pred_mask    BoolTensor (bs, output_len), filled with 1 when we need to predict a word
            y            LongTensor (pred_mask.sum(), )
        """
        x = output[pred_mask.unsqueeze(-1).expand_as(output)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores.astype("float32"), y, reduction="mean")
        return scores, loss

    def generate(self, memory, memory_key_padding_mask=None, max_len=200, sample_temperature=None):
        """
        For evaluation/testing only.
        Inputs:
            memory:                  Tensor (bs, memory_len, dim)
            memory_key_padding_mask: Optional[BoolTensor] (bs, memory_len)
        Output:
            generated:               LongTensor(bs, cur_len)
                                     e.g. <BOS> W1 W2 W3 <EOS> <PAD>
                                          <BOS> W1 W2 W3   W4  <EOS>
            gen_len:                 LongTensor(bs)
                                     e.g. [5, 6]

        """
        bs = memory.size(0)
        if self.config.kv_cache:
            memory = memory.transpose(0, 1)  # (memory_len, bs, dim)

        # generated sentences
        generated = paddle.full([max_len, bs], self.pad_index, dtype="int64")
        generated[0] = paddle.full([bs], self.bos_index, dtype="int64")

        # current position / max lengths / length of generated sentences / unfinished sentences
        cache = None
        cur_len = 1
        gen_len = paddle.ones([bs], dtype="int64")  # (bs, )
        unfinished_sents = paddle.ones([bs], dtype="int64")  # (bs, )

        # generation loop
        while cur_len < max_len:  # max length of generation
            tgt = generated[:cur_len]  # (cur_len, bs)
            tgt = self.word_embeddings(tgt)
            if self.config.kv_cache:
                if self.positional_embedding is not None:
                    tgt = self.positional_embedding(tgt, batch_first=False)  # (output_len, bs, dim)
            else:
                tgt = tgt.transpose([1, 0, 2])  # (bs, cur_len, dim)
                if self.positional_embedding is not None:
                    tgt = self.positional_embedding(tgt, batch_first=True)

            if self.config.kv_cache:
                decoded, cache = self.transformer_decoder(
                    tgt=tgt, memory=memory, cache=cache
                )  # (cur_len, bs, dim)
                scores = self.proj(decoded[-1])  # (bs, n_words)
            else:
                tgt_mask = _generate_square_subsequent_mask(tgt.size(1), dtype=tgt.dtype)
                decoded = self.transformer_decoder(
                    tgt=tgt, memory=memory, tgt_mask=tgt_mask
                )  # (bs, cur_len, dim)
                scores = self.proj(decoded[:, -1])  # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = paddle.topk(scores, 1, axis=1)[1].squeeze(1)
            else:
                next_words = paddle.multinomial(
                    F.softmax(scores.astype("float32") / sample_temperature, axis=1), num_samples=1
                ).squeeze(1)
            # assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len = gen_len + unfinished_sents
            unfinished_sents = unfinished_sents * (next_words != self.eos_index).astype("int64")
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max().item() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1] = paddle.where(
                unfinished_sents.astype("bool"),
                paddle.full_like(generated[-1], self.eos_index),
                generated[-1],
            )
        generated = generated[:cur_len].transpose(0, 1)  # (bs, cur_len)
        return generated, gen_len


"""
Transformer Fusion Module

"""


class TransformerFusion(nn.Module):
    """
    Fusion Transformer
    """

    def __init__(self, config, num_types=2):
        super().__init__()

        self.config = config
        self.dim = config.dim_emb

        if config.n_layer == 0:
            self.transformer_encoder = None
        else:
            if config.get("custom_encoder", 0):
                norm = resolve_norm(config.get("norm", "layer"))

                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=config.rotary,
                        custom_attn=config.custom_attn,
                        norm=norm,
                    ),
                    num_layers=config.n_layer,
                    norm=norm(config.dim_emb) if config.norm_first else None,
                    config=config,
                )
            else:
                self.transformer_encoder = CustomTransformerEncoder(
                    CustomTransformerEncoderLayer(
                        d_model=config.dim_emb,
                        nhead=config.n_head,
                        dim_feedforward=config.dim_ffn,
                        dropout=config.dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=config.norm_first,
                        rotary=False,
                        custom_attn=False,
                        norm=nn.LayerNorm,
                    ),
                    num_layers=config.n_layer,
                    norm=nn.LayerNorm(config.dim_emb) if config.norm_first else None,
                    config=None,
                )

        if config.type_embeddings:
            self.type_embeddings = Embedding(num_types, self.dim)
        else:
            self.type_embeddings = None

    def forward(self, x0, x1, key_padding_mask0=None, key_padding_mask1=None):
        """
        x0: Tensor (bs, slen0, dim)
        x1: Tensor (bs, slen1, dim)
        key_padding_mask0: Optional[BoolTensor] (bs, slen0)           (True for positions that should be ignored)
        key_padding_mask1: Optional[BoolTensor] (bs, slen1)
        """

        bs = x0.size(0)

        if self.type_embeddings is not None:
            type0 = paddle.zeros([1, 1], dtype="int64")
            type1 = paddle.ones([1, 1], dtype="int64")
            x0 = x0 + self.type_embeddings(type0).expand(x0.shape)
            x1 = x1 + self.type_embeddings(type1).expand(x1.shape)

        x = paddle.cat([x0, x1], axis=1)  # (bs, slen0+slen1, dim)

        if key_padding_mask0 is None and key_padding_mask1 is None:
            fused_mask = None
        else:
            if key_padding_mask0 is None:
                key_padding_mask0 = paddle.zeros([bs, x0.size(1)], dtype="bool")
            if key_padding_mask1 is None:
                key_padding_mask1 = paddle.zeros([bs, x1.size(1)], dtype="bool")
            fused_mask = paddle.cat([key_padding_mask0, key_padding_mask1], axis=1)  # (bs, slen0+slen1)

        if self.transformer_encoder is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=fused_mask)

        return x, fused_mask  # (bs, slen0+slen1, dim)
