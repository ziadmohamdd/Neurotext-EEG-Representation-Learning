"""Synthetic template-coordinate fixtures, not research observations."""
from dataclasses import replace
import json

import mne
import numpy as np
import pytest

from eeg_decoding.data.dataset import load_signal, read_manifest, write_manifest
from eeg_decoding.data.labram_inputs import derive_mapping, prepare_inputs


def template_audit():
    pos = mne.channels.make_standard_montage('GSN-HydroCel-129').get_positions()['ch_pos']
    electrodes = [dict(label=label, X=float(xyz[1]*100), Y=float(-xyz[0]*100), Z=float(xyz[2]*100))
                  for label,xyz in pos.items()]
    return {'errors':[], 'recordings':[{'channels':list(pos), 'electrodes':electrodes,
                                       'sha256':'a'*64, 'source':'synthetic_header.mat'}]}


def test_template_mapping_is_bijective_and_preserves_vertex():
    channels, mapping, report = derive_mapping(template_audit())
    assert len(channels) == len(mapping) == len(set(mapping.values())) == 19
    assert mapping['Cz'] == 'CZ'
    assert max(row['angular_error_degrees'] for row in report['mapping_errors']) <= 12
    assert report['recordings'][0]['max_residual_m'] < 1e-10


@pytest.mark.parametrize('change', ['missing', 'flipped', 'wrong_order', 'partial'])
def test_template_audit_rejects_unverified_geometry(change):
    audit = template_audit()
    if change == 'missing':
        audit['recordings'][0]['electrodes'][0]['X'] = None
    elif change == 'flipped':
        for electrode in audit['recordings'][0]['electrodes']:
            electrode['Y'] *= -1
    elif change == 'wrong_order':
        audit['recordings'][0]['electrodes'].reverse()
    else:
        audit['errors'] = [{'error':'failed header'}]
    with pytest.raises(ValueError):
        derive_mapping(audit)


def test_adaptation_uses_same_real_samples_for_baselines(records, tmp_path):
    audit = template_audit()
    audit_path = tmp_path/'coordinates.json'
    audit_path.write_text(json.dumps(audit), encoding='utf-8')
    channels = audit['recordings'][0]['channels']
    inputs = []
    for i, n in enumerate([199, 200, 401, 1801]):
        x = np.arange(len(channels)*n, dtype=np.float32).reshape(len(channels),n)
        path = tmp_path/f'input{i}.npz'
        np.savez_compressed(path, eeg=x)
        inputs.append(replace(records[i], signal_path=str(path), channels=channels, sfreq=200,
                              provenance={'units':'uV', 'metadata_sources':[{'sha256':'a'*64}]}, quality={'old':1}))
    manifest = tmp_path/'input.jsonl'
    write_manifest(manifest, inputs)
    output = tmp_path/'adapted'
    report = prepare_inputs(manifest, audit_path, output)
    assert report['n_kept'] == 3 and report['n_excluded'] == 1
    kept = read_manifest(output/'manifest.jsonl')
    assert [r.sentence_id for r in kept] == [r.sentence_id for r in inputs[1:]]
    assert [load_signal(r).shape[-1] for r in kept] == [200,400,1800]
    for r, old in zip(kept, inputs[1:]):
        index = [channels.index(c) for c in r.channels]
        np.testing.assert_array_equal(load_signal(r), load_signal(old)[index, :load_signal(r).shape[1]])
        assert r.provenance['labram_adaptation']['source_n_samples'] == load_signal(old).shape[1]
        assert r.quality == {'previous_source_quality':{'old':1}}
    smoke = read_manifest(output/'smoke.jsonl')
    assert [load_signal(r).shape[-1] for r in smoke] == [200,1800]
    with pytest.raises(FileExistsError):
        prepare_inputs(manifest, audit_path, output)
    bad = [replace(inputs[1], provenance={'units':'uV', 'metadata_sources':[{'sha256':'b'*64}]})]
    write_manifest(tmp_path/'bad.jsonl', bad)
    with pytest.raises(ValueError, match='not covered'):
        prepare_inputs(tmp_path/'bad.jsonl', audit_path, tmp_path/'bad_output')


def test_pretrained_extraction_preserves_existing_artifacts(tmp_path):
    from argparse import Namespace
    from eeg_decoding.pretrained_workflow import extract_features
    output = tmp_path/'features'
    output.mkdir()
    (output/'metadata.json').write_text('{"status":"extracting"}')
    with pytest.raises(FileExistsError):
        extract_features(Namespace(output=output))
