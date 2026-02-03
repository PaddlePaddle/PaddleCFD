import copy

import numpy as np
import paddle


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class ModelWrapper(paddle.nn.Module):
    """"""

    def __init__(
        self,
        env=None,
        embedder=None,
        encoder=None,
        decoder=None,
        beam_type="search",
        beam_length_penalty=1,
        beam_size=1,
        beam_early_stopping=True,
        max_generated_output_len=200,
        beam_temperature=1.0,
    ):
        super().__init__()
        self.env = env
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.beam_type = beam_type
        self.beam_early_stopping = beam_early_stopping
        self.max_generated_output_len = max_generated_output_len
        self.beam_size = beam_size
        self.beam_length_penalty = beam_length_penalty
        self.beam_temperature = beam_temperature
        # PaddlePaddle: parameters() 返回list，需要用iter()包装
        self.device = next(iter(self.embedder.parameters())).device

    @paddle.no_grad()
    def forward(self, input, hints):
        """
        x: bags of sequences (B, T)
        """
        env = self.env
        embedder, encoder, decoder = self.embedder, self.encoder, self.decoder
        assert "units" in env.params.use_hints
        if "units" in env.params.use_hints:
            units = hints[0]
        if env.params.use_hints == "units":
            hints[0] = [[] for _ in range(len(hints[0]))]
        B, T = len(input), max([len(xi) for xi in input])
        outputs = [[], [], []]
        for chunk in chunks(
            np.arange(B),
            max(
                1,
                min(
                    int(10000 / T),
                    int(100000 / self.beam_size / self.max_generated_output_len),
                ),
            ),
        ):
            hint = [[_hints[idx] for idx in chunk] for _hints in hints]
            x, x_len = embedder([input[idx] for idx in chunk], hint)
            encoded = encoder("fwd", x=x, lengths=x_len, causal=False).transpose(0, 1)
            bs = encoded.shape[0]
            unit = [units[idx] for idx in chunk]
            x_unit = [
                [env.equation_encoder.units_encode(u) for u in un[:-1]] for un in unit
            ]
            x_unit = env.word_to_idx(x_unit, unit_input=True)
            (
                generations,
                generations_len,
                generations_units,
                word_perplexity,
                units_perplexity,
            ) = decoder.generate(
                src_enc=encoded,
                src_len=x_len,
                decode_physical_units=env.params.decode_physical_units,
                max_len=self.max_generated_output_len,
                sample_temperature=None,
                units=x_unit,
            )
            slen = generations.shape[0]
            generations = generations.unsqueeze(-1).view(slen, bs, 1)
            generations = generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
            if generations_units is not None:
                generations_units = generations_units.unsqueeze(-1).view(slen, bs, 1, 5)
                generations_units = (
                    generations_units.transpose(0, 1)
                    .transpose(1, 2)
                    .reshape((bs, 1, -1))
                    .cpu()
                    .tolist()
                )
            else:
                generations_units = [[None] for _ in range(bs)]
            word_perplexity = word_perplexity.unsqueeze(-1).view(bs, 1).cpu().tolist()
            if units_perplexity is not None:
                units_perplexity = (
                    units_perplexity.unsqueeze(-1).view(bs, 1).cpu().tolist()
                )
            else:
                units_perplexity = [[None] for _ in range(bs)]
            generations = [
                list(
                    env.equation_encoder.decode(
                        lst=[env.equation_id2word[int(term)] for term in hyp1[1:-1]],
                        xy_units=[units[idx] for idx in chunk][i],
                        decode_physical_units=env.params.decode_physical_units,
                        units=[env.equation_id2word[int(term)] for term in hyp2[5:-5]]
                        if hyp2 is not None
                        else None,
                        lable_unit_fn=env.generator.label_units,
                    )
                    for hyp1, hyp2 in zip(generations[i], generations_units[i])
                )
                for i in range(bs)
            ]
            for i in range(bs):
                idx_to_remove = [(generations[i][j] is None) for j in range(1)]
                generations[i] = [
                    x for j, x in enumerate(generations[i]) if not idx_to_remove[j]
                ]
                word_perplexity[i] = [
                    x for j, x in enumerate(word_perplexity[i]) if not idx_to_remove[j]
                ]
                units_perplexity[i] = [
                    x for j, x in enumerate(units_perplexity[i]) if not idx_to_remove[j]
                ]
            if self.beam_type == "search":
                raise NotImplementedError()
                _, _, search_generations = decoder.generate_beam(
                    encoded,
                    x_len,
                    beam_size=self.beam_size,
                    length_penalty=self.beam_length_penalty,
                    max_len=self.max_generated_output_len,
                    early_stopping=self.beam_early_stopping,
                )
                search_generations = [
                    sorted(
                        [hyp for hyp in search_generations[i].hyp],
                        key=lambda s: s[0],
                        reverse=True,
                    )
                    for i in range(bs)
                ]
                search_generations = [
                    list(
                        filter(
                            lambda x: x is not None,
                            [
                                env.idx_to_infix(
                                    hyp.cpu().tolist()[1:],
                                    is_float=False,
                                    str_array=False,
                                )
                                for _, hyp in search_generations[i]
                            ],
                        )
                    )
                    for i in range(bs)
                ]
                for i in range(bs):
                    generations[i].extend(search_generations[i])
            elif self.beam_type == "sampling":
                num_samples = self.beam_size
                encoded = (
                    encoded.unsqueeze(1)
                    .expand((bs, num_samples) + encoded.shape[1:])
                    .contiguous()
                    .view((bs * num_samples,) + encoded.shape[1:])
                )
                x_len = x_len.unsqueeze(1).expand(bs, num_samples).contiguous().view(-1)
                sampling_x_unit = [
                    u for u in copy.deepcopy(unit) for _ in range(num_samples)
                ]
                (
                    sampling_generations,
                    sampling_generations_len,
                    sampling_generations_units,
                    sampling_word_perplexity,
                    sampling_units_perplexity,
                ) = decoder.generate(
                    src_enc=encoded,
                    src_len=x_len,
                    decode_physical_units=env.params.decode_physical_units,
                    max_len=self.max_generated_output_len,
                    sample_temperature=env.params.beam_temperature,
                    units=sampling_x_unit,
                )
                slen = sampling_generations.shape[0]
                sampling_generations = sampling_generations.unsqueeze(-1).view(
                    slen, bs, num_samples
                )
                sampling_generations = (
                    sampling_generations.transpose(0, 1).transpose(1, 2).cpu().tolist()
                )
                if sampling_generations_units is not None:
                    sampling_generations_units = sampling_generations_units.unsqueeze(
                        -2
                    ).view(slen, bs, num_samples, 5)
                    sampling_generations_units = (
                        sampling_generations_units.transpose(0, 1)
                        .transpose(1, 2)
                        .reshape((bs, 1, -1))
                        .cpu()
                        .tolist()
                    )
                else:
                    sampling_generations_units = [
                        [None for _ in range(num_samples)] for _ in range(bs)
                    ]
                sampling_word_perplexity = (
                    sampling_word_perplexity.view(bs, num_samples).cpu().tolist()
                )
                if sampling_units_perplexity is not None:
                    sampling_units_perplexity = (
                        sampling_units_perplexity.view(bs, num_samples).cpu().tolist()
                    )
                else:
                    sampling_units_perplexity = [
                        [None for _ in range(num_samples)] for _ in range(bs)
                    ]
                sampling_generations = [
                    list(
                        env.equation_encoder.decode(
                            lst=[
                                env.equation_id2word[int(term)] for term in hyp1[1:-1]
                            ],
                            xy_units=[units[idx] for idx in chunk][i],
                            decode_physical_units=env.params.decode_physical_units,
                            units=[
                                env.equation_id2word[int(term)] for term in hyp2[5:-5]
                            ]
                            if hyp2 is not None
                            else None,
                            lable_unit_fn=env.generator.label_units,
                        )
                        for hyp1, hyp2 in zip(
                            sampling_generations[i], sampling_generations_units[i]
                        )
                    )
                    for i in range(bs)
                ]
                for i in range(bs):
                    idx_to_remove = [
                        (sampling_generations[i][j] is None) for j in range(num_samples)
                    ]
                    sampling_generations[i] = [
                        x
                        for j, x in enumerate(sampling_generations[i])
                        if not idx_to_remove[j]
                    ]
                    sampling_word_perplexity[i] = [
                        x
                        for j, x in enumerate(sampling_word_perplexity[i])
                        if not idx_to_remove[j]
                    ]
                    sampling_units_perplexity[i] = [
                        x
                        for j, x in enumerate(sampling_units_perplexity[i])
                        if not idx_to_remove[j]
                    ]
                    generations[i].extend(sampling_generations[i])
                    word_perplexity[i].extend(sampling_word_perplexity[i])
                    units_perplexity[i].extend(sampling_units_perplexity[i])
                outputs[0].extend(generations)
                outputs[1].extend(word_perplexity)
                outputs[2].extend(units_perplexity)
            else:
                raise NotImplementedError
        return outputs
