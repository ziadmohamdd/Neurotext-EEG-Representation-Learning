"""Label-independent relative spectral features and rank-based class decoding."""
import numpy as np


def relative_mean(words):
    if words.ndim != 2 or words.shape[1] != 840:
        raise ValueError("Expected word x (8 bands * 105 channels)")
    valid = np.isfinite(words).all(1)
    if not np.all(valid | ~np.isfinite(words).any(1)):
        raise ValueError("Partial word missingness is outside this protocol")
    x = words[valid]
    if not len(x):
        return np.full(840, np.nan)
    z = (np.sign(x) * np.log1p(np.abs(x))).reshape(-1, 8, 105)
    z = z - z.mean(2, keepdims=True)
    return z.mean(0).reshape(840)


def score_ranks(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 2 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Invalid finite class scores")
    truth = scores[np.arange(len(scores)), labels, None]
    return 1 + (scores > truth).sum(1) + .5 * ((scores == truth).sum(1) - 1)


def choose(rows):
    best = max(r["macro_recall@1"] for r in rows)
    tied = [r for r in rows if r["macro_recall@1"] >= best - 1e-12]
    return min(tied, key=lambda r: (r["weight"], -r["alpha"]))
