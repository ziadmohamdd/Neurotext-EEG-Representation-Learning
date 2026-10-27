import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..data.dataset import sentence_key
from ..models.projection import alignment_loss
from .trainer import Trainer

class TextCache:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            self.keys, self.embeddings = f["keys"].tolist(), f["embeddings"].astype(np.float32)
            self.metadata = json.loads(str(f["metadata"].item()))
        self.index = {k: i for i, k in enumerate(self.keys)}
        if len(self.keys) != len(self.index) or len(self.keys) != len(self.embeddings):
            raise ValueError("Duplicate/misaligned sentence cache")
        if not np.isfinite(self.embeddings).all() or np.any(np.linalg.norm(self.embeddings, axis=1) < 1e-8):
            raise ValueError("Invalid text embeddings")
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def get(self, keys):
        return self.embeddings[[self.index[k] for k in keys]]

    @classmethod
    def build(cls, records, path, model_name, revision=None, batch_size=16):
        texts = {r.key: r.text for r in records}
        keys = sorted(texts)
        if Path(path).exists():
            cached = cls(path)
            if cached.keys == keys and cached.metadata.get("model") == model_name and cached.metadata.get("requested_revision") == revision:
                return cached
            raise FileExistsError("Existing sentence cache differs; use a new output path")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, revision=revision, device="cpu")
        vectors = model.encode([texts[k] for k in keys], batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        resolved = getattr(model[0].auto_model.config, "_commit_hash", None)
        metadata = {"model": model_name, "requested_revision": revision, "resolved_revision": resolved,
                    "sentence_hash_rule": "NFKC/casefold/whitespace", "n_sentences": len(keys)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.asarray(keys), embeddings=vectors, metadata=json.dumps(metadata))
        return cls(path)

def train_alignment(model, train_loader, val_loader, text_cache, config, device, output, metadata, resume=None):
    def objective(m, b):
        target = torch.from_numpy(text_cache.get(b["keys"])).to(device)
        return alignment_loss(m(b["eeg"], b["mask"]), target, b["keys"])
    trainer = Trainer(model, config, device, output)
    return trainer.fit(train_loader, val_loader, objective, metadata, resume)
