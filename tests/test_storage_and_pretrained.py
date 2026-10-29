from dataclasses import replace
import os
from pathlib import Path
import pytest
import torch
from eeg_decoding.data.dataset import write_manifest,read_manifest
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.utils.storage import stage_manifest

def test_stage_checksum_path_independent(records,tmp_path):
    source=tmp_path/"source.jsonl"
    write_manifest(source,records)
    target=stage_manifest(source,tmp_path/"staged")
    loaded=read_manifest(target)
    assert manifest_digest(loaded)==manifest_digest(records)
    assert loaded[0].signal_path!=records[0].signal_path
    again=stage_manifest(source,tmp_path/"staged")
    assert manifest_digest(read_manifest(again))==manifest_digest(records)
    with open(loaded[0].signal_path,"ab") as f:
        f.write(b"unexpected bytes")
    assert manifest_digest(read_manifest(again))!=manifest_digest(records)
    stage_manifest(source,tmp_path/"staged")
    assert manifest_digest(read_manifest(again))==manifest_digest(records)

@pytest.mark.skipif(not os.environ.get("EEG_LABRAM_CACHE"),reason="optional official-weight integration; no network download during unit tests")
def test_official_pretrained_weights_on_synthetic_input():
    from eeg_decoding.models.pretrained_eeg import FrozenLaBraM
    mapping={c:c for c in ["C3","CZ","C4"]}
    model=FrozenLaBraM(os.environ["EEG_LABRAM_CACHE"],list(mapping),mapping)
    x=torch.randn(2,3,600)
    mask=torch.ones(2,600,dtype=torch.bool)
    mask[0,400:]=False
    z=model(x,mask)
    assert z.shape==(2,200) and z.isfinite().all()
    assert not any(p.requires_grad for p in model.parameters())
    torch.testing.assert_close(z[0],model(x[:1,:,:400])[0])
    with pytest.raises(ValueError,match="one second"):
        model(x[:1,:,:100])


@pytest.mark.skipif(not os.environ.get("EEG_LABRAM_CACHE"),reason="optional official-weight extraction on synthetic fixtures")
def test_pretrained_cache_extraction(records,config,tmp_path):
    import argparse,json,yaml
    from eeg_decoding.pretrained_workflow import extract_features,FeatureCache
    path=tmp_path/"manifest.jsonl"
    write_manifest(path,records[:2])
    mapping=tmp_path/"mapping.json"
    mapping.write_text(json.dumps({c:c for c in records[0].channels}))
    cfg=tmp_path/"config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    args=argparse.Namespace(manifest=str(path),config=str(cfg),cache=os.environ["EEG_LABRAM_CACHE"],
        channel_mapping=str(mapping),output=str(tmp_path/"features"),include_controls=True,seed=42)
    extract_features(args)
    features=FeatureCache(args.output)
    features.verify(records[:2],False,42)
    assert len(features.metadata["variants"])==14
    assert features.matrix(records[:2]).shape==(2,200)
    with pytest.raises(ValueError,match="seed differs"):
        features.verify(records[:2],False,43)
    reused_args = argparse.Namespace(**dict(vars(args), output=str(tmp_path/'gaussian_only'),
                                            include_controls=False, include_gaussian=True,
                                            reuse_real_features=args.output))
    extract_features(reused_args)
    reused = FeatureCache(reused_args.output)
    assert reused.metadata['variants'] == ['real','gaussian']
    assert (Path(args.output)/'real.npy').read_bytes() == (Path(reused_args.output)/'real.npy').read_bytes()
    import numpy as np
    np.testing.assert_array_equal(reused.matrix(records[:2], 'gaussian'), features.matrix(records[:2], 'gaussian'))


@pytest.mark.parametrize('mismatch', ['mapping','max_samples','observation_ids','manifest_digest','weights_sha256'])
def test_real_feature_reuse_rejects_incompatible_inputs(tmp_path,mismatch):
    import json
    import numpy as np
    from eeg_decoding.pretrained_workflow import verify_real_reuse
    expected=dict(model='official_LaBraM_base',commit='test_commit',weights_sha256='test_weight_hash',
                  manifest_digest='test_data_hash',observation_ids=['one','two'],quality_aware=False,
                  mapping={'E2':'F8'},seed=42,units='uV_divided_by_100',max_samples=None)
    (tmp_path/'metadata.json').write_text(json.dumps(dict(expected,status='complete',variants=['real'])))
    np.save(tmp_path/'real.npy',np.ones((2,200),np.float32))
    path, checksum = verify_real_reuse(tmp_path,expected)
    assert path == tmp_path/'real.npy' and len(checksum)==64
    bad = dict(expected)
    bad[mismatch] = 'incompatible'
    with pytest.raises(ValueError,match=f'reuse mismatch: {mismatch}'):
        verify_real_reuse(tmp_path,bad)
    np.save(tmp_path/'real.npy',np.full((2,200),np.nan,np.float32))
    with pytest.raises(ValueError,match='invalid shape, dtype or values'):
        verify_real_reuse(tmp_path,expected)


def test_notebook_schema_and_cells():
    import nbformat
    root=Path(__file__).resolve().parents[1]
    notebooks=list((root/"notebooks").glob("*.ipynb"))
    assert len(notebooks)==7
    for path in notebooks:
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=="code":
                compile(cell.source,str(path),"exec")
                assert not cell.outputs
