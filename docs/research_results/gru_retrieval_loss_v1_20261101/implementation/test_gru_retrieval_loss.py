import torch
from torch.nn import functional as F
import pytest
from eeg_decoding.evaluation.gru_retrieval_loss import candidate_cross_entropy


def test_loss_matches_full_candidate_reference_and_has_finite_gradient():
    torch.manual_seed(42)
    predictions = torch.randn(4, 5, requires_grad=True)
    candidates = torch.randn(7, 5)
    labels = torch.tensor([0, 2, 2, 5])
    actual = candidate_cross_entropy(predictions, candidates, labels)
    expected = F.cross_entropy(F.normalize(predictions) @ F.normalize(candidates).T / .07, labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(predictions.grad).all() and predictions.grad.abs().sum() > 0


def test_duplicate_target_observations_do_not_duplicate_candidates():
    candidates = torch.eye(3)
    p = torch.tensor([[1., .1, .1], [1., .1, .1]])
    labels = torch.tensor([0, 0])
    torch.testing.assert_close(candidate_cross_entropy(p, candidates, labels),
                               candidate_cross_entropy(p[:1], candidates, labels[:1]))


def test_candidate_reordering_and_label_remap_preserve_loss():
    p = torch.tensor([[1., 2., 3.], [2., 1., 3.]])
    c = torch.eye(3)
    order = torch.tensor([2, 0, 1])
    inverse = torch.argsort(order)
    y = torch.tensor([0, 2])
    torch.testing.assert_close(candidate_cross_entropy(p, c, y), candidate_cross_entropy(p, c[order], inverse[y]))


def test_invalid_inputs_fail():
    with pytest.raises(ValueError):
        candidate_cross_entropy(torch.ones(2, 3), torch.ones(3, 4), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        candidate_cross_entropy(torch.ones(2, 3), torch.eye(3), torch.tensor([0, 1]), 0)
