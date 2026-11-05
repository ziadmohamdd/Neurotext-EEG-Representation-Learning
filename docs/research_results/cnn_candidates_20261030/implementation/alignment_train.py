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
    scope = config.get('alignment_candidate_scope','batch')
    if scope not in {'batch','training_unique'}:
        raise ValueError('alignment_candidate_scope must be batch or training_unique')
    candidates, candidate_keys = None, None
    if scope == 'training_unique':
        candidate_keys = sorted(metadata['splits']['splits']['train']['sentence_keys'])
        candidates = torch.from_numpy(text_cache.get(candidate_keys)).to(device)
    if 'alignment_candidate_scope' in config:
        initial = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            initial.update(name.encode())
            initial.update(value.detach().cpu().contiguous().numpy().tobytes())
        metadata['initial_model_sha256'] = initial.hexdigest()
        metadata['alignment_objective'] = dict(forward_candidates=scope,
            candidate_keys=candidate_keys,
            candidate_keys_sha256=hashlib.sha256(json.dumps(candidate_keys).encode()).hexdigest() if candidate_keys else None,
            reverse_term='unchanged_symmetric_batch_multi_positive',temperature=.07,
            model_selection='common_held_out_subject_batch_alignment_loss',
            training_bank_scope='training sentence texts only; no validation/test-only keys')
    def objective(m, b):
        target = torch.from_numpy(text_cache.get(b["keys"])).to(device)
        # Both arms use exactly the original batch loss on validation. Their
        # different training losses must not select checkpoints on different scales.
        return alignment_loss(m(b["eeg"], b["mask"]), target, b["keys"],
                              candidates=candidates if m.training else None,
                              candidate_keys=candidate_keys if m.training else None)
    trainer = Trainer(model, config, device, output)
    return trainer.fit(train_loader, val_loader, objective, metadata, resume)
