import paddle


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    if len(sequences) == 0:
        return []
    max_len = max(seq.shape[0] for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        if pad_len > 0:
            pad_shape = [pad_len] + list(seq.shape[1:])
            pad_tensor = paddle.full(shape=pad_shape, fill_value=padding_value, dtype=seq.dtype)
            seq = paddle.concat([seq, pad_tensor], axis=0)
        padded.append(seq)
    axis = 0 if batch_first else 1
    return paddle.stack(padded, axis=axis)
