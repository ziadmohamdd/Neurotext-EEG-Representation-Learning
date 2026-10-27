import numpy as np
import pytest
import torch
from eeg_decoding.models.eeg_cnn import EEGCNN
from eeg_decoding.models.ssl_encoder import MaskedEEG
from eeg_decoding.models.classical import eeg_features,ridge_model
from eeg_decoding.models.projection import alignment_loss
from eeg_decoding.training.checkpoints import save_checkpoint,load_checkpoint
from eeg_decoding.data.dataset import EEGDataset,collate_variable

@pytest.mark.parametrize("quality",[False,True])
def test_forward_padding_invariance(records,quality):
    dataset=EEGDataset(records,max_samples=None)
    model=EEGCNN(3,128,quality_aware=quality).eval()
    batch=collate_variable([dataset[0],dataset[9]])
    with torch.no_grad():
        together=model(batch["eeg"],batch["mask"])
        alone=model(dataset[0]["eeg"][None])
    assert together.shape==(2,128)
    torch.testing.assert_close(together[0],alone[0])


def test_ssl_loss_backprop():
    model=MaskedEEG(3)
    x=torch.randn(2,3,200)
    mask=torch.ones(2,200,dtype=torch.bool)
    loss=model.loss(x,mask)
    assert torch.isfinite(loss) and loss>0
    loss.backward()
    assert model.encoder.features[0].weight.grad.abs().sum()>0


@pytest.mark.parametrize("length", [2, 3, 19, 20, 21, 40])
def test_ssl_short_observations_keep_context_and_ignore_padding(length):
    model = MaskedEEG(3)
    x = torch.ones(1, 3, length + 7, requires_grad=True)
    mask = torch.arange(length + 7)[None] < length
    seen = []
    handle = model.encoder.features.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    loss = model.loss(x, mask, generator=torch.Generator().manual_seed(42))
    handle.remove()
    assert torch.isfinite(loss)
    corrupted = seen[0]
    assert corrupted.shape[-1] == length
    hidden = (corrupted == 0).all(dim=1)
    assert 0 < hidden.sum() < length
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :, length:].count_nonzero() == 0
    assert model.decoder.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mask", [[[True, False, True]], [[True, False, False]]])
def test_ssl_rejects_invalid_validity_masks(mask):
    with pytest.raises(ValueError, match="prefix-valid"):
        MaskedEEG(3).loss(torch.ones(1, 3, 3), torch.tensor(mask))


def test_alignment_multi_positive():
    z=torch.eye(3,requires_grad=True)
    loss=alignment_loss(z,z,["same","same","different"])
    assert torch.isfinite(loss)
    loss.backward()
    with pytest.raises(ValueError):
        alignment_loss(z,z,["same"]*3)


def test_classical_features():
    features=eeg_features(np.random.default_rng(0).normal(size=(3,300)).astype(np.float32),200)
    assert features.shape==(30,) and np.isfinite(features).all()
    model=ridge_model().fit(np.stack([features,features+1,features+2]),np.eye(3))
    assert model.predict(features[None]).shape==(1,3)


def test_checkpoint(records,tmp_path):
    model=EEGCNN(3)
    optimizer=torch.optim.AdamW(model.parameters())
    path=tmp_path/"model.pt"
    save_checkpoint(path,model,optimizer,epoch=3)
    expected=model(torch.randn(2,3,100))
    restored=EEGCNN(3)
    meta=load_checkpoint(path,restored,restore_rng=True)
    assert meta["epoch"]==3
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b)


def test_ridge_correlated_features_matches_double_precision_svd():
    import warnings
    from scipy.linalg import LinAlgWarning
    from sklearn.preprocessing import StandardScaler
    rng=np.random.default_rng(9)
    latent=rng.normal(size=(64,4))
    # Near-identical/duplicated columns and a constant mimic correlated EEG statistics.
    x=np.column_stack([latent,latent[:,0],latent[:,0]+1e-7*latent[:,1],np.ones(64)]).astype(np.float32)
    y=(latent@rng.normal(size=(4,3))).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('error',LinAlgWarning)
        model=ridge_model(.1).fit(x,y)
        prediction=model.predict(x)
    assert np.isfinite(prediction).all() and prediction.dtype==np.float64
    scaler=StandardScaler().fit(x.astype(np.float64))
    standardized=scaler.transform(x.astype(np.float64))
    u,s,vt=np.linalg.svd(standardized,full_matrices=False)
    centered=y.astype(np.float64)-y.astype(np.float64).mean(0)
    reference=(standardized@vt.T)@((s/(s*s+.1))[:,None]*(u.T@centered))+y.astype(np.float64).mean(0)
    np.testing.assert_allclose(prediction,reference,rtol=1e-10,atol=1e-10)
