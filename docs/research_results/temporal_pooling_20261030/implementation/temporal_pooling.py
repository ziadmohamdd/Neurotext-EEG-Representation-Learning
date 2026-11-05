"""Matched initialization for the fixed one-bin versus four-bin comparison."""
import hashlib
import torch
from .eeg_cnn import EEGCNN
from .projection import EEGTextModel


def matched_pooling_models(channels, eeg_dim=128, text_dim=384, seed=42, quality_aware=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        baseline = EEGTextModel(EEGCNN(channels, eeg_dim, quality_aware=quality_aware), eeg_dim, text_dim)
        variant = EEGTextModel(EEGCNN(channels, eeg_dim, quality_aware=quality_aware, temporal_segments=4), eeg_dim, text_dim)
    variant.encoder.features.load_state_dict(baseline.encoder.features.state_dict())
    variant.projection.load_state_dict(baseline.projection.state_dict())
    with torch.no_grad():
        # Flatten is channel-major, so repeat_interleave, not repeat, is required.
        variant.encoder.head.weight.copy_(baseline.encoder.head.weight.repeat_interleave(4, dim=1) / 4)
        variant.encoder.head.bias.copy_(baseline.encoder.head.bias)
    return baseline, variant


def state_hash(model, names=None):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if names is None or name in names:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialization_report(baseline, variant):
    left, right = baseline.state_dict(), variant.state_dict()
    shared = [k for k in left if left[k].shape == right[k].shape]
    assert all(torch.equal(left[k], right[k]) for k in shared)
    return {"global_parameters": sum(p.numel() for p in baseline.parameters()),
            "four_segment_parameters": sum(p.numel() for p in variant.parameters()),
            "shared_tensor_names": shared, "shared_tensor_sha256": state_hash(baseline, shared),
            "shared_tensors_identical": True, "global_initial_sha256": state_hash(baseline),
            "four_segment_initial_sha256": state_hash(variant),
            "head_initialization": "channel-major repeat_interleave of global head weights / 4; identical bias"}
