import importlib.util
from pathlib import Path

import pytest
import torch
from eeg_decoding.models.eeg_cnn import EEGCNN
from eeg_decoding.models.temporal_pooling import matched_pooling_models, initialization_report


def test_matched_initialization_and_parameter_increase():
    baseline, variant = matched_pooling_models(105)
    report = initialization_report(baseline, variant)
    assert report['shared_tensors_identical']
    assert report['four_segment_parameters'] - report['global_parameters'] == 32 * 3 * 128
    assert torch.equal(variant.encoder.head.weight, baseline.encoder.head.weight.repeat_interleave(4, dim=1) / 4)
    assert report['global_initial_sha256'] == 'd4702d0f3cd3617b14b6a5eed91f3d6b01a9b9dc5698de9f2ce794d88d12d5e3'


def test_equal_bin_initial_function_and_segment_gradients():
    baseline, variant = matched_pooling_models(3)
    x = torch.randn(2,3,80)
    torch.testing.assert_close(baseline(x),variant(x),rtol=1e-5,atol=1e-6)
    variant(x)[:,0].sum().backward()
    grad = variant.encoder.head.weight.grad.reshape(128,32,4)
    assert torch.isfinite(grad).all()
    assert not torch.allclose(grad[:,:,0],grad[:,:,3])


@pytest.mark.parametrize('length', [2, 3, 19, 20, 21, 80])
def test_four_segment_masks_ignore_padding(length):
    model=EEGCNN(3,temporal_segments=4).eval()
    x=torch.randn(1,3,length+13)
    mask=torch.arange(length+13)[None] < length
    torch.testing.assert_close(model(x,mask),model(x[:,:,:length]))


def test_ordered_bins_can_distinguish_equal_global_means():
    h=torch.arange(8,dtype=torch.float32)[None,None].repeat(1,32,1)
    model=EEGCNN(32,embedding_dim=1,temporal_segments=4)
    model.features=torch.nn.Identity()
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.zero_();model.head.weight[0,0]=1
    assert torch.equal(h.mean(-1),h.flip(-1).mean(-1))
    assert not torch.equal(model(h),model(h.flip(-1)))


def test_global_default_preserves_original_pooling_and_state_shapes():
    model=EEGCNN(3).eval();x=torch.randn(1,3,83)
    torch.testing.assert_close(model(x),model.head(model.features(x).mean(-1)))
    assert model.head.weight.shape==(128,32)
    with pytest.raises(ValueError):EEGCNN(3,temporal_segments=2)


def test_validation_screen_rejects_improvement_shared_by_controls():
    spec=importlib.util.spec_from_file_location('pool_runner',Path(__file__).parents[1]/'scripts/run_temporal_pooling_comparison.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    reports=[]
    for pooling in ['global','four_segment']:
        for control in ['real','gaussian','shuffled']:
            value=(.1 if pooling=='global' else .2) + (.01 if control=='real' else 0)
            reports.append({'pooling':pooling,'control':control,'metrics':{'mrr':value,'recall@1':value,'recall@5':value,'recall@10':value},'per_subject':{'s1':{'mrr':value}}})
    outcome=runner.validation_gate(reports)
    assert not outcome['useful_validation_improvement']
    assert not outcome['test_evaluated']
