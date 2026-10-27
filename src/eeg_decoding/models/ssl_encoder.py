import torch
from torch import nn
from .eeg_cnn import EEGCNN

class MaskedEEG(nn.Module):
    def __init__(self, channels, embedding_dim=128, quality_aware=False):
        super().__init__()
        self.encoder = EEGCNN(channels, embedding_dim, quality_aware=quality_aware)
        self.decoder = nn.Conv1d(32, channels, 1)

    def loss(self, x, mask, generator=None, fraction=.3, block=20):
        if not 0 < fraction < 1:
            raise ValueError("Mask fraction must lie in (0,1)")
        if not isinstance(block, int) or block < 1:
            raise ValueError("Temporal block size must be a positive integer")
        losses = []
        for sample, valid in zip(x, mask):
            n = int(valid.sum())
            if n < 2 or not valid[:n].all() or valid[n:].any():
                raise ValueError("Expected prefix-valid mask with at least two samples")
            # Short observations still need visible context and a masked target.
            # Reduce block width only when the default would hide the whole signal;
            # never pad a target or silently drop an observation.
            sample_block = min(block, max(1, n // 2)) if n <= block else block
            nblocks = (n + sample_block - 1) // sample_block
            scores = torch.rand(nblocks, generator=generator, device=x.device)
            chosen = scores.argsort()[:max(1, min(nblocks - 1, round(nblocks * fraction))) ]
            hidden = torch.zeros(nblocks, dtype=torch.bool, device=x.device)
            hidden[chosen] = True
            hidden = hidden.repeat_interleave(sample_block)[:n]
            target = sample[None, :, :n]
            corrupted = target.masked_fill(hidden[None, None], 0)
            reconstructed = self.decoder(self.encoder.sequence_features(corrupted))
            losses.append((reconstructed - target).square()[:, :, hidden].mean())
        return torch.stack(losses).mean()
