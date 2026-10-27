import numpy as np

def signal_quality(x):
    finite = np.isfinite(x)
    safe = np.where(finite, x, 0)
    return {"finite_fraction": float(finite.mean()), "rms_uv": float(np.sqrt(np.mean(safe ** 2))),
            "peak_uv": float(np.abs(safe).max()), "flat_channel_fraction": float((safe.std(axis=1) < 1e-6).mean()),
            "duration_seconds_samples": int(x.shape[-1])}

def channel_reliability(x):
    # No target labels or test-distribution fitting; attenuate channels with large transients.
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    peak = (x - x.mean(dim=-1, keepdim=True)).abs().amax(dim=-1, keepdim=True)
    return (3 * std / peak.clamp_min(1e-6)).clamp(max=1)
