import numpy as np

def sample_bounds(onset, duration, sfreq, n_samples):
    if not np.isfinite([onset, duration, sfreq]).all() or onset < 0 or duration <= 0 or sfreq <= 0:
        raise ValueError("Invalid sentence onset/duration/sampling rate")
    start, stop = int(round(onset * sfreq)), int(round((onset + duration) * sfreq))
    if start < 0 or stop > n_samples or stop - start < 2:
        raise ValueError("Sentence interval outside recording or too short")
    return start, stop

def harmonize(x, source_channels, target_channels):
    if len(set(target_channels)) != len(target_channels) or not target_channels:
        raise ValueError("Target channels must be unique and nonempty")
    missing = set(target_channels) - set(source_channels)
    if missing:
        raise ValueError(f"Missing channels: {sorted(missing)}")
    return x[[source_channels.index(c) for c in target_channels]].astype(np.float32)
