"""Synthetic optimization checks; these are not scientific EEG results."""
import json

import pytest

from eeg_decoding.diagnostics import select_tiny_training_set, overfit_diagnostic
from eeg_decoding.training.checkpoints import load_checkpoint


def test_diagnostic_uses_only_one_training_subject(records):
    selected,splits=select_tiny_training_set(records,42,8)
    assert len(selected)==len({r.key for r in selected})==8
    assert len({r.subject_key for r in selected})==1
    assert all(r in splits['train'] for r in selected)
    assert not ({r.subject_key for r in selected} & {r.subject_key for r in splits['test']+splits['val']})
    again,_=select_tiny_training_set(list(reversed(records)),42,8)
    assert [r.signal_path for r in again]==[r.signal_path for r in selected]


def test_diagnostic_optimizes_and_stays_out_of_research_exports(records,text_cache,config,tmp_path):
    out=tmp_path/'diagnostic'
    report=overfit_diagnostic(records,text_cache.path,config,out,steps=40,n_sentences=4)
    assert report['status']=='diagnostic_completed' and not report['generalization_evidence']
    assert report['final']['loss'] < report['initial']['loss']
    assert report['normalizer']['subjects']
    assert report['synthetic'] and 'metrics' not in report
    assert load_checkpoint(out/'diagnostic.pt')['stage']=='diagnostic_overfit'
    audit=json.loads((out/'audit.json').read_text())
    assert len(audit['records'])==4 and not audit['identical_signal_pairs']
    with pytest.raises(FileExistsError):
        overfit_diagnostic(records,text_cache.path,config,out,steps=1,n_sentences=4)


@pytest.mark.parametrize('n',[1,17])
def test_diagnostic_requires_small_distinct_target_pool(records,n):
    with pytest.raises(ValueError,match='2 to 16'):
        select_tiny_training_set(records,42,n)
