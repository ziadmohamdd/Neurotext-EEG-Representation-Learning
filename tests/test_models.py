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
