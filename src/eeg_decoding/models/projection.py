import torch
from torch import nn
from torch.nn import functional as F

class EEGTextModel(nn.Module):
    def __init__(self, encoder, eeg_dim, text_dim, frozen=False):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        self.projection = nn.Sequential(nn.Linear(eeg_dim, text_dim), nn.LayerNorm(text_dim))
        if frozen:
            self.encoder.requires_grad_(False)

    def forward(self, x, mask=None):
        if self.frozen:
            self.encoder.eval()
            with torch.no_grad():
                z = self.encoder(x, mask)
        else:
            z = self.encoder(x, mask)
        return F.normalize(self.projection(z), dim=-1)

def alignment_loss(predicted, targets, keys, temperature=.07):
    """Symmetric multi-positive cross entropy: duplicate text is never a false negative."""
    if len(set(keys)) < 2:
        raise ValueError("Alignment needs at least two distinct sentences per batch")
    logits = F.normalize(predicted.float(), dim=-1) @ F.normalize(targets.float(), dim=-1).T / temperature
    positives = torch.tensor([[a == b for b in keys] for a in keys], device=logits.device)
    def direction(scores, positive):
        return -(torch.logsumexp(scores.masked_fill(~positive, -torch.inf), dim=1) - torch.logsumexp(scores, dim=1)).mean()
    return (direction(logits, positives) + direction(logits.T, positives.T)) / 2
