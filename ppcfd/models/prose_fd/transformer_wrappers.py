import einops
import paddle

"""
Final model wrappers. 
"""
from logging import getLogger

from .embedder import get_embedder
from .transformer import (DataOperatorDecoder, TransformerDataEncoder,
                          TransformerFusion, TransformerSymbolEncoder)

logger = getLogger()


class PROSE_2to1(paddle.nn.Module):
    """
    Wrapper for the PROSE model (2to1).
    """

    def __init__(self, config, symbol_env, x_num, max_output_dim, output_len=1):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = x_num
        self.max_output_dim = max_output_dim
        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.data_encoder = TransformerDataEncoder(config.data_encoder)
        self.symbol_encoder = TransformerSymbolEncoder(
            config.symbol_encoder, symbol_env.equation_id2word
        )
        self.fusion = TransformerFusion(config.fusion)
        if config.embedder.type == "fourier":
            p = config.data_decoder.patch_num_output
            space_len = p * (p // 2 + 1)
        else:
            space_len = None
        self.data_decoder = DataOperatorDecoder(
            config.data_decoder, output_len, space_len
        )

    def summary(self):
        s = "\n"
        s += f"""	Embedder:        {sum([p.numel() for p in self.embedder.parameters() if not p.stop_gradient]):,}
"""
        s += f"""	Data Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if not p.stop_gradient]):,}
"""
        s += f"""	Symbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if not p.stop_gradient]):,}
"""
        s += f"""	Fusion:          {sum([p.numel() for p in self.fusion.parameters() if not p.stop_gradient]):,}
"""
        s += f"\tData Decoder:    {sum([p.numel() for p in self.data_decoder.parameters() if not p.stop_gradient]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        data_input,
        input_times,
        output_times,
        symbol_input,
        symbol_padding_mask=None,
        **kwargs,
    ):
        """
        Inputs:
            data_input:          Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:         Tensor     (bs/1, input_len, 1)
            output_times:        Tensor     (bs/1, output_len, 1)

            symbol_input:        LongTensor           (bs, symbol_len)
            symbol_padding_mask: Optional[BoolTensor] (bs, symbol_len)     symbol padding mask, positions with True are padding

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        if self.config.get("carry_last_frame", 0):
            last_frame = data_input[:, -1:].clone()
        bs = data_input.size(0)
        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, input_len, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = input_len * patch_num * patch_num
        """
        data_input = self.embedder.encode(data_input, input_times)
        """
        Step 2: Encode + Fusion
            data_input:   Tensor     (bs, data_len, dim)
            symbol_input: LongTensor (bs, symbol_len)
        """
        data_encoded = self.data_encoder(data_input)
        symbol_encoded = self.symbol_encoder(
            symbol_input, src_key_padding_mask=symbol_padding_mask
        )
        fused, fused_mask = self.fusion(
            x0=data_encoded,
            x1=symbol_encoded,
            key_padding_mask0=None,
            key_padding_mask1=symbol_padding_mask,
        )
        """
        Step 3: Decode data
        """
        query_emb = self.data_decoder.get_query_emb(output_times)
        if query_emb.size(0) == 1:
            query_emb = query_emb.expand(bs, -1, -1)
        data_output = self.data_decoder(
            src=fused, query_emb=query_emb, src_key_padding_mask=fused_mask
        )
        data_output = self.embedder.decode(data_output)
        if self.config.get("carry_last_frame", 0):
            data_output = data_output + last_frame
        return data_output


class PROSE_1to1(paddle.nn.Module):
    """
    Wrapper for the PROSE model (1to1).
    """

    def __init__(self, config, symbol_env, x_num, max_output_dim, output_len=1):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = x_num
        self.max_output_dim = max_output_dim
        self.embedder = get_embedder(config.embedder, x_num, max_output_dim)
        self.data_encoder = TransformerDataEncoder(config.data_encoder)
        self.data_decoder = DataOperatorDecoder(config.data_decoder, output_len)

    def summary(self):
        s = "\n"
        s += f"""	Embedder:        {sum([p.numel() for p in self.embedder.parameters() if not p.stop_gradient]):,}
"""
        s += f"""	Data Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if not p.stop_gradient]):,}
"""
        s += f"\tData Decoder:    {sum([p.numel() for p in self.data_decoder.parameters() if not p.stop_gradient]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.fwd(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, input_times, output_times, **kwargs):
        """
        Inputs:
            data_input:          Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:         Tensor     (bs/1, input_len, 1)
            output_times:        Tensor     (bs/1, output_len, 1)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_input.size(0)
        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, input_len, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = input_len * patch_num * patch_num
        """
        data_input = self.embedder.encode(data_input, input_times)
        """
        Step 2: Encode
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_encoded = self.data_encoder(data_input)
        """
        Step 3: Decode data
        """
        query_emb = self.data_decoder.get_query_emb(output_times)
        if query_emb.size(0) == 1:
            query_emb = query_emb.expand(bs, -1, -1)
        data_output = self.data_decoder(
            src=data_encoded, query_emb=query_emb, src_key_padding_mask=None
        )
        data_output = self.embedder.decode(data_output)
        return data_output
