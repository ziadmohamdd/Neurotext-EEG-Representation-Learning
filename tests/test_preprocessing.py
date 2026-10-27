import numpy as np
import pytest
from eeg_decoding.preprocessing.normalization import ChannelNormalizer
from eeg_decoding.preprocessing.filtering import filter_resample
from eeg_decoding.preprocessing.artifacts import handle_artifacts
from eeg_decoding.preprocessing.quality import signal_quality
from eeg_decoding.data.dataset import load_signal

def test_training_only_normalization(records):
    train=[r for r in records if r.subject=="0"]
    norm=ChannelNormalizer().fit(train,{train[0].subject_key})
    all_samples=np.concatenate([load_signal(r) for r in train],axis=1)
    np.testing.assert_allclose(norm.mean[:,0],all_samples.mean(1),atol=1e-6)
    np.testing.assert_allclose(norm.std[:,0],all_samples.std(1),atol=1e-6)
    restored=ChannelNormalizer.from_state(norm.state_dict())
    np.testing.assert_allclose(restored.transform(all_samples).mean(1),0,atol=1e-6)
    with pytest.raises(ValueError):
        ChannelNormalizer().fit(records,{train[0].subject_key})

def test_filter_resample():
    fs=500
    t=np.arange(2000)/fs
    x=np.array([np.sin(2*np.pi*10*t)+np.sin(2*np.pi*100*t)],dtype=np.float32)
    y=filter_resample(x,fs,target_sfreq=200)
    assert y.shape==(1,800) and y.dtype==np.float32
    f=np.fft.rfftfreq(800,1/200)
    spectrum=np.abs(np.fft.rfft(y[0]))
    assert abs(f[spectrum.argmax()]-10)<.5
    with pytest.raises(ValueError):
        filter_resample(x,50)

def test_already_filtered():
    x=np.random.default_rng(0).normal(size=(3,200)).astype(np.float32)
    np.testing.assert_array_equal(x,filter_resample(x,200,already_filtered=True))

def test_artifact_quality():
    x=np.array([[0,300,0]],dtype=np.float32)
    with pytest.raises(ValueError):
        handle_artifacts(x)
    assert handle_artifacts(x,policy="clip").max()==200
    assert signal_quality(x)["peak_uv"]==300
