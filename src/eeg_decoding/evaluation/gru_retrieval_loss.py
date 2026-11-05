"""Full training-candidate loss for a frozen GRU loss-only comparison."""
import torch
from torch.nn import functional as F


def candidate_cross_entropy(prediction, candidates, labels, temperature=.07):
    if prediction.ndim != 2 or candidates.ndim != 2 or prediction.shape[1] != candidates.shape[1]:
        raise ValueError("Expected matching sentence embedding widths")
    if labels.ndim != 1 or len(labels) != len(prediction) or len(candidates) < 2 or temperature <= 0:
        raise ValueError("Invalid candidate labels or temperature")
    if not torch.isfinite(prediction).all() or not torch.isfinite(candidates).all():
        raise ValueError("Nonfinite embeddings")
    logits = F.normalize(prediction, dim=-1) @ F.normalize(candidates, dim=-1).T / temperature
    return F.cross_entropy(logits, labels)
