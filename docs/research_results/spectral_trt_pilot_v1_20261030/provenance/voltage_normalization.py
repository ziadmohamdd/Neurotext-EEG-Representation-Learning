import numpy as np
from ..data.dataset import load_signal

class ChannelNormalizer:
    """Streaming population moments, fit exclusively to a supplied training split."""
    def fit(self, records, permitted_subjects):
        if not records or not set(r.subject_key for r in records) <= set(permitted_subjects):
            raise ValueError("Normalizer received non-training subjects or empty data")
        mean, m2, count = None, None, 0
        for r in records:
            x = load_signal(r).astype(np.float64)
            n, mu, var = x.shape[-1], x.mean(axis=1), x.var(axis=1)
            if mean is None:
                mean, m2, count = mu, var * n, n
            else:
                delta = mu - mean
                m2 += var * n + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
        self.mean = mean.astype(np.float32)[:, None]
        self.std = np.sqrt(m2 / count).clip(1e-6).astype(np.float32)[:, None]
        self.subjects = sorted(permitted_subjects)
        return self

    def transform(self, x):
        if x.shape[0] != len(self.mean):
            raise ValueError("Normalization channel mismatch")
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "subjects": self.subjects}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.mean, obj.std = np.array(state["mean"], np.float32), np.array(state["std"], np.float32)
        obj.subjects = state["subjects"]
        return obj
