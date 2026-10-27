import numpy as np

def gaussian_input(x, rng):
    # Per-observation/channel amplitude matched, without sentence labels.
    return (rng.normal(size=x.shape) * x.std(axis=-1, keepdims=True)).astype(np.float32)

def shuffled_keys(keys, seed=42):
    """Derange sentence *classes*, so repeated observations keep a consistent wrong label."""
    unique = sorted(set(keys))
    if len(unique) < 2:
        raise ValueError("Shuffled-pair control requires two sentence classes")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    shift = int(rng.integers(1, len(unique)))
    mapping = {key: unique[(i + shift) % len(unique)] for i, key in enumerate(unique)}
    return [mapping[k] for k in keys]
