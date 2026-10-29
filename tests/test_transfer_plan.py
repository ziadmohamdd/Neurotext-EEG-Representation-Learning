"""Synthetic fixtures: preparation must not silently authorize strict transfer."""
from dataclasses import replace
import json

import numpy as np
import pytest

from eeg_decoding.data.dataset import read_manifest,write_manifest
from eeg_decoding.data.transfer_plan import prepare_plan
from eeg_decoding.training.alignment_train import TextCache


def inputs(records,text_cache,tmp_path):
    a=[replace(r,provenance={k:v for k,v in r.provenance.items() if k!='person_id'}) for r in records[:30]]
    b=[replace(r,dataset='zuco2',provenance={k:v for k,v in r.provenance.items() if k!='person_id'}) for r in records[30:]]
    paths=[tmp_path/'a.jsonl',tmp_path/'b.jsonl',tmp_path/'ta.npz',tmp_path/'tb.npz']
    write_manifest(paths[0],a)
    write_manifest(paths[1],b)
    metadata=dict(synthetic=True,model='SYNTHETIC_TEST_ONLY',resolved_revision='fixture',
                  requested_revision=None,sentence_hash_rule='NFKC/casefold/whitespace')
    for path in paths[2:]:
        np.savez_compressed(path,keys=np.array(text_cache.keys),embeddings=text_cache.embeddings,metadata=json.dumps(metadata))
    return paths


def test_preparation_keeps_identity_gate_and_original_signal_references(records,text_cache,tmp_path):
    paths=inputs(records,text_cache,tmp_path)
    output=tmp_path/'plan'
    report=prepare_plan(*paths,output)
    assert report['status']=='blocked_pending_participant_identity'
    assert report['n_observations']==60 and report['n_unique_sentences']==10
    assert len(report['directions'])==2
    assert all(d['status']=='not_executed' and not d['identity_gate_passes'] for d in report['directions'])
    assert all('identity audit missing' in d['blocking_reason'] for d in report['directions'])
    combined=read_manifest(report['manifest'])
    assert all('person_id' not in r.provenance for r in combined)
    assert {r.signal_path for r in combined}=={r.signal_path for r in records}
    assert {p.name for p in output.iterdir()}=={'combined.jsonl','sentence_embeddings.npz','plan.json'}
    merged=TextCache(report['text_cache'])
    np.testing.assert_allclose(merged.get(text_cache.keys),text_cache.embeddings,rtol=1e-6,atol=1e-7)
    with pytest.raises(FileExistsError):
        prepare_plan(*paths,output)


@pytest.mark.parametrize('change',['model','vectors','channels'])
def test_incompatible_inputs_are_rejected_before_writing_plan(records,text_cache,tmp_path,change):
    paths=inputs(records,text_cache,tmp_path)
    if change=='channels':
        b=read_manifest(paths[1]);b=[replace(r,channels=['X','Y','Z']) for r in b]
        write_manifest(paths[1],b)
        match='Harmonize'
    else:
        cache=TextCache(paths[3]);metadata=cache.metadata
        vectors=cache.embeddings.copy()
        if change=='model':metadata['model']='OTHER_SYNTHETIC_MODEL'
        else:vectors[0]*=-1
        np.savez_compressed(paths[3],keys=np.array(cache.keys),embeddings=vectors,metadata=json.dumps(metadata))
        match='Text model' if change=='model' else 'embeddings disagree'
    with pytest.raises(ValueError,match=match):
        prepare_plan(*paths,tmp_path/'out')
    assert not (tmp_path/'out').exists()
