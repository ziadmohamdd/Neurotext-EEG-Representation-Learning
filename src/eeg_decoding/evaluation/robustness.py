import numpy as np

def corrupt(x, kind, severity, rng):
    if severity < 0 or kind not in {"additive_noise", "channel_dropout", "temporal_mask"}:
        raise ValueError("Unknown corruption or negative severity")
    if kind != "additive_noise" and severity > 1:
        raise ValueError("Drop/mask severity must lie in [0,1]")
    y = x.copy()
    if kind == "additive_noise":
        y += (rng.normal(size=x.shape) * x.std(axis=-1, keepdims=True) * severity).astype(np.float32)
    elif kind == "channel_dropout":
        indices = rng.choice(x.shape[0], round(x.shape[0] * severity), replace=False)
        y[indices] = 0
    else:
        length = round(x.shape[-1] * severity)
        start = int(rng.integers(0, x.shape[-1] - length + 1))
        y[:, start:start+length] = 0
    return y
