import paddle

from ppcfd.models.prose_fd.paddle_utils import *


def pad_sequence(sequences, batch_first=True, padding_value=0):
    max_len = max(sequence.shape[0] for sequence in sequences)
    batch_size = len(sequences)
    out_shape = (
        [batch_size, max_len, *sequences[0].shape[1:]]
        if batch_first
        else [max_len, batch_size, *sequences[0].shape[1:]]
    )
    padded = paddle.full(out_shape, padding_value, dtype=sequences[0].dtype)
    for idx, sequence in enumerate(sequences):
        length = sequence.shape[0]
        if batch_first:
            padded[idx, :length] = sequence
        else:
            padded[:length, idx] = sequence
    return padded


def get_padding_mask(lengths, max_len=None, pad_right=1):
    """
    Input:
        lengths:           LongTensor (bs, )  length of each example
        max_len:           Optional[int]      if None, max_len = lengths.max()
    Output:
        key_padding_mask:  BoolTensor (bs, max_len)    (positions with value True are padding)
    """
    if max_len is None:
        max_len = lengths._max().item()
    bs = lengths.size(0)
    if pad_right:
        key_padding_mask = paddle.arange(max_len, device=lengths.device).expand(
            bs, max_len
        ) >= lengths.unsqueeze(1)
    else:
        key_padding_mask = paddle.arange(max_len, device=lengths.device).expand(
            bs, max_len
        ) < max_len - lengths.unsqueeze(1)
    return key_padding_mask


def get_data_mask(lengths, max_len=None):
    """
    Input:
        lengths:           LongTensor (bs, )  length of each example
        max_len:           Optional[int]      if None, max_len = lengths.max()
    Output:
        data_mask:         Tensor (bs, max_len)    (positions with value 0 are padding)
    """
    if max_len is None:
        max_len = lengths._max().item()
    bs = lengths.size(0)
    mask = paddle.arange(max_len, device=lengths.device).expand(
        bs, max_len
    ) < lengths.unsqueeze(1)
    return mask.float()


def custom_collate(
    max_data_dim,
    padding_idx=-100,
    tie_fields=True,
    mixed_length=0,
    input_len=-1,
    pad_right=1,
):
    def my_collate(batch):
        res = {}
        keys = batch[0].keys()
        for k in keys:
            if k == "data" and tie_fields:
                lst = []
                dims = []
                if not mixed_length:
                    for d in batch:
                        cur_data = d[k]
                        data_dim = cur_data.size(-1)
                        dims.append(data_dim)
                        diff = max_data_dim - data_dim
                        if diff > 0:
                            cur_data = paddle.compat.nn.functional.pad(
                                cur_data, (0, diff), "constant"
                            )
                        lst.append(cur_data)
                    data_dims = paddle.LongTensor(dims)
                    res["data_mask"] = get_data_mask(data_dims, max_data_dim)[
                        :, None, None, None, :
                    ]
                else:
                    lens = []
                    for d in batch:
                        cur_data = d[k]
                        data_dim = cur_data.size(-1)
                        dims.append(data_dim)
                        diff = max_data_dim - data_dim
                        if diff > 0:
                            cur_data = paddle.compat.nn.functional.pad(
                                cur_data, (0, diff), "constant"
                            )
                        data_len = cur_data.size(0)
                        lens.append(data_len - input_len)
                        diff = mixed_length - data_len
                        if diff > 0:
                            cur_data = paddle.compat.nn.functional.pad(
                                cur_data, (0, 0, 0, 0, 0, 0, 0, diff), "constant"
                            )
                        lst.append(cur_data)
                    data_mask_dim = get_data_mask(
                        paddle.LongTensor(dims), max_data_dim
                    )[:, None, None, None, :]
                    data_mask_len = get_data_mask(
                        paddle.LongTensor(lens), mixed_length - input_len
                    )[:, :, None, None, None]
                    res["data_mask"] = data_mask_dim * data_mask_len
                res[k] = paddle.io.dataloader.collate.default_collate_fn(lst)
            elif k == "symbol_input":
                if pad_right:
                    symbols = [d[k] for d in batch]
                    lengths = paddle.LongTensor([l.size(0) for l in symbols])
                    symbols_pad = pad_sequence(
                        symbols, batch_first=True, padding_value=padding_idx
                    )
                else:
                    symbols = [d[k].flip(axis=[0]) for d in batch]
                    lengths = paddle.LongTensor([l.size(0) for l in symbols])
                    symbols_pad = pad_sequence(
                        symbols, batch_first=True, padding_value=padding_idx
                    ).flip(axis=[1])
                res["symbol_mask"] = get_padding_mask(lengths, pad_right=pad_right)
                res["symbol_input"] = symbols_pad
            else:
                res[k] = paddle.io.dataloader.collate.default_collate_fn(
                    [d[k] for d in batch]
                )
        return res

    return my_collate


if __name__ == "__main__":
    data = [
        paddle.LongTensor([1, 2, 3]),
        paddle.LongTensor([1, 2]),
        paddle.LongTensor([1]),
    ]
    dict = [{"symbol_input": text} for text in data]
    collate = custom_collate(10)
    res = collate(dict)
    print(res["symbol_mask"])
    print(res["symbol_input"])
