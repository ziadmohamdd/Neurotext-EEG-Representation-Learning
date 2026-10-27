"""Frozen official LaBraM-base adapter. Never silently falls back to random weights."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import urllib.request
import argparse
import sys
import numpy as np
import torch
from torch import nn

COMMIT = "c431221e6cfd23dbfa9950e0180682fb322b0548"
WEIGHTS_SHA256 = "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"
BASE = f"https://raw.githubusercontent.com/935963004/LaBraM/{COMMIT}/"
SOURCE_HASHES = {
    "modeling_finetune.py": "7b3514c9a661ebd417a2dd9834a9f2fab0a2e518c8212c73ebebaaa165c29d88",
    "upstream_utils.py": "c5ad729bd57ab31e04572f426d94f7fb164ef0fbfd1f7076da6bf8dfbd833e32",
    "LICENSE": "bc3f03e2c9071fb40507f5c575d95eeaa98d07c54c54fb2855942b2a729adc71",
}

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def acquire_official(cache_dir):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for remote, filename in [("modeling_finetune.py", "modeling_finetune.py"), ("utils.py", "upstream_utils.py"),
                             ("LICENSE", "LICENSE"), ("checkpoints/labram-base.pth", "labram-base.pth")]:
        target = cache / filename
        if target.exists():
            continue
        temp = target.with_suffix(target.suffix + ".partial")
        with urllib.request.urlopen(BASE + remote, timeout=120) as response, temp.open("wb") as out:
            while part := response.read(1024 * 1024):
                out.write(part)
        if filename.endswith(".pth") and file_sha256(temp) != WEIGHTS_SHA256:
            temp.unlink()
            raise ValueError("Official weight checksum mismatch")
        temp.replace(target)
    if file_sha256(cache / "labram-base.pth") != WEIGHTS_SHA256:
        raise ValueError("Cached LaBraM weights failed integrity check")
    for filename, expected in SOURCE_HASHES.items():
        if file_sha256(cache / filename) != expected:
            raise ValueError(f"Official LaBraM source checksum mismatch: {filename}")
    return cache

class FrozenLaBraM(nn.Module):
    embedding_dim = 200
    def __init__(self, cache_dir, channels, channel_mapping, sfreq=200):
        super().__init__()
        if sfreq != 200:
            raise ValueError("LaBraM requires 200 Hz")
        if not channel_mapping or not set(channels) <= channel_mapping.keys():
            raise ValueError("Provide an electrode-position-audited source-to-10/20 channel mapping")
        mapped = [channel_mapping[c].upper() for c in channels]
        if len(set(mapped)) != len(mapped):
            raise ValueError("Channel mapping must be one-to-one")
        cache = Path(cache_dir)
        if file_sha256(cache / "labram-base.pth") != WEIGHTS_SHA256:
            raise ValueError("LaBraM checkpoint SHA256 mismatch")
        for filename, expected in SOURCE_HASHES.items():
            if file_sha256(cache / filename) != expected:
                raise ValueError(f"LaBraM source integrity check failed: {filename}")
        # Read the official electrode vocabulary without importing unrelated training dependencies.
        tree = ast.parse((cache / "upstream_utils.py").read_text())
        vocabulary = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "standard_1020" for t in node.targets):
                vocabulary = ast.literal_eval(node.value)
        if vocabulary is None or any(c not in vocabulary for c in mapped):
            raise ValueError("Mapped electrode absent from official LaBraM vocabulary")
        self.input_chans = [0] + [vocabulary.index(c) + 1 for c in mapped]
        spec = importlib.util.spec_from_file_location("eeg_official_labram", cache / "modeling_finetune.py")
        module = sys.modules.get(spec.name)
        if module is None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        self.backbone = module.labram_base_patch200_200(num_classes=0, init_values=.1,
                                                       qkv_bias=False, use_abs_pos_emb=True,
                                                       use_mean_pooling=False)
        numpy_core = np._core if hasattr(np, '_core') else np.core
        with torch.serialization.safe_globals([argparse.Namespace, (numpy_core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
                                               np.dtype, type(np.dtype('float64'))]):
            state = torch.load(cache / "labram-base.pth", map_location="cpu", weights_only=True)
        state = state.get("model", state)
        student = {k[len("student."):]: v for k, v in state.items() if k.startswith("student.")}
        wanted = self.backbone.state_dict()
        # Drop only pretraining heads/tokens that are absent in the official inference backbone.
        state = {k: v for k, v in student.items() if k in wanted}
        if set(state) != set(wanted):
            raise ValueError(f"Incomplete pretrained backbone: {sorted(set(wanted)-set(state))}")
        self.backbone.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device)
        features = []
        for sample, valid in zip(x, mask):
            n = int(valid.sum())
            if n < 200:
                raise ValueError("LaBraM needs at least one second; do not pad invented EEG")
            chunks = []
            # Non-overlapping <=8 second windows bound attention memory. Last partial second dropped.
            for start in range(0, n - 199, 1600):
                length = min(1600, n - start) // 200 * 200
                patches = sample[:, start:start+length].reshape(1, sample.shape[0], length // 200, 200)
                tokens = self.backbone(patches / 100., input_chans=self.input_chans, return_patch_tokens=True)
                chunks.append((tokens.mean(1), length))
            features.append(sum(z * length for z, length in chunks) / sum(length for _, length in chunks))
        return torch.cat(features)
