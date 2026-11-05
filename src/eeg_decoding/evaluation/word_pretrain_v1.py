"""Small word-target stage and explicit projection-only transfer."""
import numpy as np
import torch
from torch import nn


def fit_vocabulary(training_label_sequences):
    vocabulary = sorted({w for row in training_label_sequences for w in row})
    if len(vocabulary) < 2:
        raise ValueError('At least two fold-training word classes required')
    return vocabulary


def target_indices(labels, vocabulary):
    lookup = {w: i for i, w in enumerate(vocabulary)}
    return np.array([lookup.get(w, -1) for w in labels], dtype=np.int64)


class WordEncoder(nn.Module):
    def __init__(self, vocabulary_size, projection=32):
        super().__init__()
        self.project = nn.Linear(841, projection)
        self.word_head = nn.Linear(projection, vocabulary_size)

    def forward(self, eeg):
        if eeg.ndim != 2 or eeg.shape[1] != 841:
            raise ValueError('Only 840 EEG coordinates plus observed-word mask allowed')
        return self.word_head(torch.tanh(self.project(eeg)))


def transfer_projection(word_model, sentence_model):
    """Copy weight and bias only. GRU/head and AdamW states never transfer."""
    sentence_model.project.load_state_dict(word_model.project.state_dict())


def index_batches(size, steps, batch_size, seed=42):
    rng = np.random.default_rng(seed)
    done = 0
    while done < steps:
        order = rng.permutation(size)
        for start in range(0, size, batch_size):
            yield order[start:start + batch_size]
            done += 1
            if done == steps:
                return


def word_ranks(logits, labels):
    """OOV is an automatic miss with reciprocal rank zero, never filtered out."""
    result = np.full(len(labels), np.inf)
    for i, y in enumerate(labels):
        if y >= 0:
            truth = logits[i, y]
            result[i] = 1 + (logits[i] > truth).sum() + .5 * ((logits[i] == truth).sum() - 1)
    return result
