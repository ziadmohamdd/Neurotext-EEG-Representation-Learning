"""Full stimulus-word sequences; missing slots and padded tails are distinct."""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def relative(words):
    finite = np.isfinite(words)
    if words.ndim != 2 or words.shape[1] != 840 or not np.all(finite.all(1) | ~finite.any(1)):
        raise ValueError("Expected complete or entirely missing 840-coordinate words")
    z = np.sign(words) * np.log1p(np.abs(words))
    z = z.reshape(-1, 8, 105)
    return (z - z.mean(2, keepdims=True)).reshape(-1, 840)


def fit_scaler(sequences):
    values = np.concatenate([relative(w)[np.isfinite(w).all(1)] for w in sequences])
    if not len(values):
        raise ValueError("No observed fold-training words")
    scale = values.std(0)
    scale[scale < 1e-8] = 1
    return {"center": values.mean(0), "scale": scale, "fit_words": np.array(len(values))}


def encode(words, scaler, condition="real_eeg", seed=42):
    valid = np.isfinite(words).all(1)
    z = np.zeros((max(1, len(words)), 841), dtype=np.float32)
    values = (relative(words)[valid] - scaler["center"]) / scaler["scale"]
    if condition == "word_order_shuffled":
        values = values[np.random.default_rng(seed).permutation(len(values))]
    if condition == "mean_eeg" and len(values):
        values[:] = values.mean(0)
    if condition != "missingness_only":
        z[np.flatnonzero(valid), :840] = values
    z[:len(words), 840] = valid
    return z


def collate(sequences, indices, device):
    lengths = torch.tensor([len(sequences[i]) for i in indices], dtype=torch.long)
    x = torch.zeros((len(indices), int(lengths.max()), 841), device=device)
    for j, i in enumerate(indices):
        x[j, :lengths[j]] = torch.as_tensor(sequences[i], device=device)
    return x, lengths


class SequenceGRU(nn.Module):
    def __init__(self, output_width=384, projection=32, hidden=32):
        super().__init__()
        self.project = nn.Linear(841, projection)
        self.gru = nn.GRU(projection, hidden, batch_first=True)
        self.head = nn.Linear(hidden, output_width)

    def forward(self, x, lengths):
        z = torch.tanh(self.project(x))
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])


def reader_folds(subjects, forbidden=("ZJS", "ZKW", "ZAB", "ZKB")):
    subjects = np.asarray(subjects)
    if set(subjects) & set(forbidden) or len(set(subjects)) != 8:
        raise ValueError("Only original eight training readers may enter CV")
    return [(s, np.flatnonzero(subjects != s), np.flatnonzero(subjects == s)) for s in sorted(set(subjects))]
