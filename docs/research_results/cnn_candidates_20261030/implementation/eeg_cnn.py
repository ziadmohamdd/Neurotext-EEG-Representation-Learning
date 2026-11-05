import torch
from torch import nn
from ..preprocessing.quality import channel_reliability

class EEGCNN(nn.Module):
    """Compact spatial mixer + depthwise temporal filters; masked adaptive pooling.

    Each true sequence is encoded separately: padding cannot alter convolutions,
    normalization or pooled embeddings. Small microbatches make this affordable.
    """
    def __init__(self, channels, embedding_dim=128, width=32, quality_aware=False):
        super().__init__()
        self.quality_aware, self.embedding_dim = quality_aware, embedding_dim
        self.features = nn.Sequential(nn.Conv1d(channels, width, 1), nn.GELU(),
                                      nn.Conv1d(width, width, 15, padding=7, groups=width),
                                      nn.GroupNorm(4, width), nn.GELU(),
                                      nn.Conv1d(width, width, 9, padding=4), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(width, embedding_dim)

    def sequence_features(self, x):
        if self.quality_aware:
            x = x * channel_reliability(x)
        return self.features(x)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device)
        out = []
        for sample, valid in zip(x, mask):
            n = int(valid.sum())
            if n < 2 or not valid[:n].all() or valid[n:].any():
                raise ValueError("Expected prefix-valid mask with at least two samples")
            h = self.sequence_features(sample[None, :, :n])
            out.append(self.head(self.pool(h).squeeze(-1)))
        return torch.cat(out)
