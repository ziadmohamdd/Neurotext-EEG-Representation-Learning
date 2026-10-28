"""Audited HydroCel template adaptation for frozen LaBraM, with matched baseline inputs."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mne
from mne.channels.montage import transform_to_head
import numpy as np
from scipy.optimize import linear_sum_assignment

from .dataset import load_signal, read_manifest, write_manifest
from .metadata import sha256_file
from ..utils.logging import write_json

TARGETS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8',
           'P7','P3','Pz','P4','P8','O1','O2']


def derive_mapping(audit):
    recordings = audit.get('recordings', [])
    if audit.get('errors') or not recordings:
        raise ValueError('Require a complete coordinate audit without header errors')
    labels = recordings[0]['channels']
    if 'Cz' not in labels or len(set(labels)) != len(labels):
        raise ValueError('Require unique source labels and the known Cz anchor')
    hydrocel = mne.channels.make_standard_montage('GSN-HydroCel-129')
    native = hydrocel.get_positions()['ch_pos']
    if any(label not in native for label in labels):
        raise ValueError('Source labels are not a HydroCel-129 subset')
    template = np.array([native[label] for label in labels])
    residuals, verified_hashes = [], []
    for recording in recordings:
        if recording['channels'] != labels:
            raise ValueError('Recording channel order differs from the audited cohort')
        electrodes = recording['electrodes']
        if [e['label'] for e in electrodes] != labels:
            raise ValueError('Coordinate labels do not match channel order')
        raw = np.array([[e.get(axis) for axis in 'XYZ'] for e in electrodes], dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError('Missing/nonfinite XYZ positions; cannot derive mapping')
        # EEGLAB: X anterior, Y left, Z superior. MNE native: X right, Y anterior.
        xyz = raw[:, [1,0,2]] * [-1,1,1]
        scale = float(np.sum(xyz*template) / np.sum(xyz*xyz))
        error = float(np.linalg.norm(xyz*scale-template, axis=1).max())
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(error) or error > 1e-4:
            raise ValueError('Stored coordinates do not match the HydroCel template within 0.1 mm')
        residuals.append({'file':Path(recording['source']).name, 'sha256':recording['sha256'],
                          'scale_to_template_m':scale, 'max_residual_m':error})
        verified_hashes.append(recording['sha256'])
    source_pos = transform_to_head(hydrocel).get_positions()['ch_pos']
    target_pos = transform_to_head(mne.channels.make_standard_montage('standard_1020')).get_positions()['ch_pos']
    a = np.array([source_pos[label] for label in labels])
    b = np.array([target_pos[label] for label in TARGETS])
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(b @ a.T, -1, 1)))
    # Preserve the named vertex rather than allowing a nearby sensor to replace it.
    source_ids = [i for i, label in enumerate(labels) if label != 'Cz']
    target_ids = [i for i, label in enumerate(TARGETS) if label != 'Cz']
    rows, cols = linear_sum_assignment(angles[np.ix_(target_ids, source_ids)])
    pairs = [(TARGETS.index('Cz'), labels.index('Cz'))]
    pairs += [(target_ids[i], source_ids[j]) for i,j in zip(rows,cols)]
    if len(pairs) != len(TARGETS) or any(angles[i,j] > 12 for i,j in pairs):
        raise ValueError('Cannot map all 19 targets one-to-one within 12 degrees')
    mapping = {labels[j]:TARGETS[i].upper() for i,j in pairs}
    channels = [label for label in labels if label in mapping]
    report = {'status':'template_verified_positional_tokens_approximate', 'mne_version':mne.__version__,
              'method':'HydroCel template match; native fiducials to head frame; unit-direction angular assignment; Cz anchor',
              'maximum_allowed_angle_degrees':12,
              'mapping_errors':[{'source':labels[j], 'target':TARGETS[i].upper(),
                                 'angular_error_degrees':float(angles[i,j])} for i,j in pairs],
              'recordings':residuals, 'verified_header_hashes':verified_hashes,
              'limitation':'Template coordinates, not individual digitization. Standard-position tokens approximate EGI positions; this is not exact electrode equivalence.'}
    return channels, mapping, report


def prepare_inputs(manifest, audit_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing adaptation artifacts are preserved')
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8-sig'))
    channels, mapping, report = derive_mapping(audit)
    records = read_manifest(manifest)
    hashes = set(report['verified_header_hashes'])
    for r in records:
        headers = r.provenance.get('metadata_sources', [])
        if not headers or any(h['sha256'] not in hashes for h in headers):
            raise ValueError('Manifest recording headers are not covered by the coordinate audit')
        if r.sfreq != 200 or r.provenance.get('units') != 'uV' or r.channels != audit['recordings'][0]['channels']:
            raise ValueError('Require matching audited channels and physical microvolts at 200 Hz')
    output.mkdir(parents=True)
    report.update(status='preparing', coordinate_audit_sha256=sha256_file(audit_path),
                  excluded=[], n_input=len(records), n_kept=0,
                  signal_recipe='Preserve upstream processing; select mapped source channels; retain complete seconds. No new bandpass/notch or normalization.',
                  baseline_requirement='Use this identical manifest and full temporal support for matched baselines')
    write_json(output/'audit.json', report)
    write_json(output/'mapping.json', mapping)
    write_json(output/'channels.json', channels)
    indices = [records[0].channels.index(c) for c in channels]
    kept = []
    for i, r in enumerate(records):
        x = load_signal(r)
        length = x.shape[1] // 200 * 200
        if length < 200:
            report['excluded'].append({'dataset':r.dataset,'subject':r.subject,'sentence_id':r.sentence_id,
                                       'n_samples':x.shape[1], 'reason':'less_than_one_complete_second'})
            continue
        identity = [r.dataset,r.subject,r.session,r.task,r.sentence_id]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        dest = output/(key+'.npz')
        if dest.exists():
            raise ValueError(f'Duplicate observation identity: {identity}')
        np.savez_compressed(dest, eeg=x[indices, :length])
        provenance = dict(r.provenance, previous_source=r.signal_path,
                          labram_adaptation={'coordinate_audit_sha256':report['coordinate_audit_sha256'],
                                             'source_n_samples':x.shape[1], 'retained_n_samples':length,
                                             'mapping':mapping, 'positional_mapping':'template_approximation'})
        kept.append(replace(r, signal_path=str(dest), channels=channels, provenance=provenance,
                            quality={'previous_source_quality':r.quality}))
        if (i+1) % 250 == 0:
            print(f'Adapted {i+1}/{len(records)} source observations', flush=True)
    if not kept:
        raise ValueError('No observations have a complete second')
    write_manifest(output/'manifest.jsonl', kept)
    smoke = [kept[0], max(kept, key=lambda r:r.provenance['labram_adaptation']['retained_n_samples'])]
    if smoke[0] == smoke[1]:
        smoke = smoke[:1]
    write_manifest(output/'smoke.jsonl', smoke)
    report.update(status='prepared', n_kept=len(kept), n_excluded=len(report['excluded']))
    write_json(output/'audit.json', report)
    print(f'Prepared {len(kept)} observations, {len(channels)} channels; excluded {len(report["excluded"])} under one second', flush=True)
    print(f'Manifest: {output/"manifest.jsonl"}', flush=True)
    print(f'Mapping: {output/"mapping.json"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--coordinate-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare_inputs(args.manifest, args.coordinate_audit, args.output)


if __name__ == '__main__':
    main()
