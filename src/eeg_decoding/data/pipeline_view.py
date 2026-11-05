"""Load-time signal fixes for the v2 pilot. Prepared signal files stay unchanged."""
import numpy as np
from scipy.signal import filtfilt, iirnotch
import torch
from torch.utils.data import DataLoader
from .dataset import collate_variable, load_signal, observation_seed
from ..evaluation.controls import gaussian_input
from ..workflow import DistinctSentenceBatches, ObservationView

ALL_READERS = "__all__"


def clean_signal(record, pipeline):
    """Apply the channel drop and mains notch that both releases must share."""
    x = load_signal(record)
    dropped = set(pipeline["drop_channels"])
    keep = [i for i, name in enumerate(record.channels) if name not in dropped]
    if len(keep) != len(record.channels) - len(dropped):
        raise ValueError("A channel to drop is absent")
    x = x[keep]
    if pipeline.get("notch_hz"):
        b, a = iirnotch(pipeline["notch_hz"], 30, fs=record.sfreq)
        if x.shape[1] <= 3 * max(len(a), len(b)):
            raise ValueError("Signal too short for the notch filter")
        x = filtfilt(b, a, x, axis=-1).astype(np.float32)
    return x


def center_crop(x, samples):
    if samples and x.shape[1] > samples:
        start = (x.shape[1] - samples) // 2
        return x[:, start:start + samples]
    return x


class ReaderNormalizer:
    """Channel moments from training rows only, shared or kept per reader."""
    def fit(self, records, pipeline, per_reader):
        self.per_reader = per_reader
        moments = {}
        for r in records:
            x = clean_signal(r, pipeline).astype(np.float64)
            name = r.subject_key if per_reader else ALL_READERS
            n, mu, var = x.shape[-1], x.mean(axis=1), x.var(axis=1)
            if name not in moments:
                moments[name] = [mu, var * n, n]
            else:
                mean, m2, count = moments[name]
                delta = mu - mean
                moments[name] = [mean + delta * n / (count + n), m2 + var * n + delta ** 2 * count * n / (count + n), count + n]
        self.mean = {k: v[0].astype(np.float32)[:, None] for k, v in moments.items()}
        self.std = {k: np.sqrt(v[1] / v[2]).clip(1e-6).astype(np.float32)[:, None] for k, v in moments.items()}
        return self

    def transform(self, x, reader):
        name = reader if self.per_reader else ALL_READERS
        if name not in self.mean:
            raise ValueError("Reader has no training statistics")
        return ((x - self.mean[name]) / self.std[name]).astype(np.float32)

    def state_dict(self):
        return {"per_reader": self.per_reader, "mean": {k: v.tolist() for k, v in self.mean.items()},
                "std": {k: v.tolist() for k, v in self.std.items()}}


class PipelineView(ObservationView):
    def __init__(self, records, normalizer, pipeline, control="real", seed=42):
        super().__init__(records, None, pipeline["window_samples"], control, None, seed)
        self.pipeline, self.reader_normalizer = pipeline, normalizer

    def __getitem__(self, index):
        r = self.records[index]
        x = center_crop(clean_signal(r, self.pipeline), self.max_samples)
        if self.control == "gaussian":
            x = gaussian_input(x, np.random.default_rng(observation_seed(r, self.seed)))
        x = self.reader_normalizer.transform(x, r.subject_key)
        item = {"eeg": torch.from_numpy(x.copy()), "key": r.key, "index": index}
        if self.training_keys is not None:
            item["key"] = self.training_keys[index]
        return item


def pipeline_loader(records, config, normalizer, pipeline, training=False, alignment=False, control="real"):
    dataset = PipelineView(records, normalizer, pipeline, control, config.get("seed", 42))
    batch = config.get("batch_size", 2)
    options = dict(num_workers=config.get("num_workers", 0), collate_fn=collate_variable,
                   pin_memory=config.get("pin_memory", True) and torch.cuda.is_available())
    if alignment:
        keys = dataset.training_keys or [r.key for r in records]
        return DataLoader(dataset, batch_sampler=DistinctSentenceBatches(keys, batch, training, config.get("seed", 42)), **options)
    return DataLoader(dataset, batch_size=batch, shuffle=training, **options)
