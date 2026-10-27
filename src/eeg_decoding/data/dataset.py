"""Versioned JSONL metadata; per-sentence compressed arrays, never a corpus tensor."""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import json
import unicodedata
import numpy as np
import torch
from torch.utils.data import Dataset

SCHEMA_VERSION = 1

def observation_seed(record, seed=42):
    identity = [record.dataset, record.subject, record.session, record.task, record.sentence_id]
    return (int(hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16], 16) + seed) % (2**63)

def sentence_key(text):
    canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not canonical:
        raise ValueError("Empty sentence text")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

@dataclass
class Record:
    dataset: str
    subject: str
    session: str
    task: str
    sentence_id: str
    text: str
    signal_path: str
    channels: list[str]
    sfreq: float
    quality: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def subject_key(self):
        # Identity reconciliation across versions must be explicit, never inferred from prefixes.
        return self.provenance.get("person_id", f"{self.dataset}:{self.subject}")

    @property
    def key(self):
        return sentence_key(self.text)

def read_manifest(path):
    path = Path(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = Record(**json.loads(line))
            if r.schema_version != SCHEMA_VERSION:
                raise ValueError("Unsupported manifest schema")
            if not Path(r.signal_path).is_absolute():
                r.signal_path = str((path.parent / r.signal_path).resolve())
            records.append(r)
    if not records:
        raise ValueError("Manifest contains no observations")
    return records

def write_manifest(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)

def load_signal(record):
    with np.load(record.signal_path, allow_pickle=False) as data:
        x = np.asarray(data["eeg"], dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != len(record.channels) or x.shape[1] < 2:
        raise ValueError(f"Invalid C x T signal: {record.signal_path}")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite signal")
    return x

class EEGDataset(Dataset):
    def __init__(self, records, normalizer=None, max_samples=1600):
        self.records, self.normalizer, self.max_samples = records, normalizer, max_samples
        if not records:
            raise ValueError("Empty dataset")
        if any(r.channels != records[0].channels or r.sfreq != records[0].sfreq for r in records):
            raise ValueError("Channels/order and sampling rate must be harmonized")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        x = load_signal(r)
        if self.max_samples and x.shape[1] > self.max_samples:
            # Deterministic center crop; same rule for all splits and models.
            start = (x.shape[1] - self.max_samples) // 2
            x = x[:, start:start + self.max_samples]
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        return {"eeg": torch.from_numpy(x.copy()), "key": r.key, "index": index}

def collate_variable(batch):
    channels = {b["eeg"].shape[0] for b in batch}
    if len(channels) != 1:
        raise ValueError("Channel mismatch")
    lengths = torch.tensor([b["eeg"].shape[1] for b in batch])
    x = torch.zeros(len(batch), next(iter(channels)), int(lengths.max()))
    mask = torch.arange(x.shape[-1])[None, :] < lengths[:, None]
    for i, b in enumerate(batch):
        x[i, :, :lengths[i]] = b["eeg"]
    return {"eeg": x, "mask": mask, "lengths": lengths, "keys": [b["key"] for b in batch],
            "indices": torch.tensor([b["index"] for b in batch])}
