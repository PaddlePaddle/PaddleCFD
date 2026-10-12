def modulate_scale_shift(x, scale, shift):
    if x.ndim == 3:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
    return x * (1 + scale) + shift


def modulate_gate(x, gate):
    if x.ndim == 3:
        gate = gate.unsqueeze(1)
    return gate * x
