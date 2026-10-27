import numpy as np

def handle_artifacts(x, threshold_uv=200., policy="reject"):
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite EEG must be rejected")
    bad = np.abs(x) > threshold_uv
    if policy == "reject" and bad.any():
        raise ValueError("Amplitude artifact exceeds threshold")
    if policy == "clip":
        return np.clip(x, -threshold_uv, threshold_uv).astype(np.float32)
    if policy not in {"reject", "flag"}:
        raise ValueError("Unknown artifact policy")
    return x
