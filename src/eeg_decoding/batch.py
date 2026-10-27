"""Prepare a complete local NR cohort incrementally, with per-subject audits and restart state."""
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from .data.acquisition import RESULT, HEADER
from .data.dataset import read_manifest, write_manifest, load_signal
from .data.metadata import read_eeg_metadata, sha256_file
from .data.validation import validate_records, manifest_digest
from .data.zuco1 import iter_mat
from .preprocessing.filtering import filter_resample
from .preprocessing.quality import signal_quality
from .preprocessing.segmentation import harmonize
from .training.alignment_train import TextCache
from .utils.logging import write_json
from .utils.plotting import signal_plot

PIPELINE_VERSION = 1


def discover(root):
    exports, headers = {}, {}
    for path in sorted(Path(root).rglob('*.mat')):
        match = RESULT.fullmatch(path.name)
        if match:
            exports.setdefault(match[1].upper(), []).append(path)
        else:
            match = HEADER.search(path.name)
            if match:
                headers.setdefault(match[1].upper(), []).append(path)
    if not exports:
        raise ValueError('No resultsSUBJECT_NR.mat exports found; download normal-reading MATLAB exports first')
    return exports, headers


def _subject_metadata(subject, paths):
    if not paths:
        raise FileNotFoundError(f'{subject}: need at least one preprocessed *_'+subject+'_NR*_EEG.mat recording for ordered channel metadata')
    metadata = [read_eeg_metadata(path) for path in paths]
    first = metadata[0]
    if any(m['sfreq'] != first['sfreq'] or m['channels'] != first['channels'] for m in metadata):
        raise ValueError(f'{subject}: supplied runs disagree in sampling rate or ordered channel labels; resolve before import')
    return first, metadata


def prepare_batch(root, dataset, output, units, target_sfreq=200, channel_policy='strict',
                  cache_text=True, text_model='sentence-transformers/all-MiniLM-L6-v2', text_revision=None, subjects=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if units not in {'uV', 'V'} or not np.isfinite(target_sfreq) or target_sfreq <= 0:
        raise ValueError('Declare valid input units and positive target sample rate')
    if channel_policy not in {'strict', 'intersection'}:
        raise ValueError('Unknown channel policy')
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'summary.json'
    exports, headers = discover(root)
    if subjects:
        requested = {s.upper() for s in subjects}
        if requested - exports.keys():
            raise ValueError(f'Missing requested exports: {sorted(requested - exports.keys())}')
        exports = {s:paths for s,paths in exports.items() if s in requested}
    report = {'status': 'running', 'dataset': dataset, 'task': 'nr', 'subjects': {},
              'input_units': units, 'target_sfreq': target_sfreq, 'channel_policy': channel_policy,
              'pipeline_version': PIPELINE_VERSION,
              'cohort_scope': 'explicitly requested subjects' if subjects else 'all locally present NR exports; not a claim that the full release was downloaded',
              'preprocessing': 'existing source preprocessing preserved; anti-alias resampling only',
              'metadata_scope': 'all locally supplied matching run headers checked; absent run headers not audited'}
    write_json(report_path, report)
    available = {}
    for subject, sources in exports.items():
        try:
            if len(sources) != 1:
                raise ValueError('Multiple exports for this subject; use one immutable release per input folder')
            first, all_metadata = _subject_metadata(subject, headers.get(subject, []))
            available[subject] = (sources[0], first, all_metadata)
        except Exception as exc:
            report['subjects'][subject] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        write_json(report_path, report)
    if not available:
        report.update(status='failed', error='No exports with usable subject-specific EEG metadata')
        write_json(report_path, report)
        return report
    first = next(iter(available.values()))[1]
    shared = set(first['channels'])
    if channel_policy == 'intersection':
        for _, meta, _ in available.values():
            shared &= set(meta['channels'])
    channels = [c for c in first['channels'] if c in shared]
    if not channels:
        raise ValueError('No shared electrode labels')
    report['channels'] = channels
    combined = []
    for subject, (source, meta, all_meta) in available.items():
        audit = {}
        subject_dir = output / subject
        subject_dir.mkdir(exist_ok=True)
        state_path, manifest = subject_dir/'state.json', subject_dir/'manifest.jsonl'
        try:
            if channel_policy == 'strict' and set(meta['channels']) != set(channels):
                raise ValueError('Electrode sets differ; resolve metadata or explicitly select --channel-policy intersection')
            signature = {'pipeline_version': PIPELINE_VERSION, 'dataset': dataset, 'source_sha256': sha256_file(source),
                         'header_sha256': [m['sha256'] for m in all_meta], 'sfreq': meta['sfreq'],
                         'input_units': units, 'target_sfreq': target_sfreq, 'channels': channels}
            digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            if state_path.exists() and manifest.exists():
                state = json.loads(state_path.read_text())
                if state.get('signature') == signature:
                    records = read_manifest(manifest)
                    if manifest_digest(records) != state['manifest_digest']:
                        raise ValueError('Previously prepared cache changed; inspect outputs before restarting')
                    combined.extend(records)
                    report['subjects'][subject] = dict(state['audit'], status='reused', manifest=str(manifest))
                    print(f'{subject}: reused {len(records)} observations', flush=True)
                    write_json(report_path, report)
                    continue
            schema = {'subject': subject, 'session': 'unspecified', 'task': 'nr', 'channels': meta['channels'],
                      'sfreq': meta['sfreq'], 'signal_field': 'rawData', 'text_field': 'content',
                      'time_axis': 1, 'units': units}
            write_json(subject_dir/'schema.json', schema)
            scratch = subject_dir/'scratch'
            processed = subject_dir/digest
            processed.mkdir(exist_ok=True)
            records = []
            print(f'{subject}: reading sentence waveforms incrementally', flush=True)
            for record in iter_mat(source, schema, scratch, dataset=dataset, skip_missing_eeg=True, audit=audit):
                original = load_signal(record)
                before = signal_quality(original)
                aligned = harmonize(original, record.channels, channels)
                signal = filter_resample(aligned, record.sfreq, target_sfreq=target_sfreq,
                                         low=None, high=None, notch=None, already_filtered=True)
                if not np.isfinite(signal).all() or signal.shape[-1] < 2:
                    raise ValueError(f'Invalid resampled signal for sentence {record.sentence_id}')
                dest = processed / (record.sentence_id + '.npz')
                np.savez_compressed(dest, eeg=signal)
                provenance = dict(record.provenance, source_sha256=signature['source_sha256'],
                                  metadata_sources=[{'sha256':m['sha256'], 'file':Path(m['source']).name} for m in all_meta],
                                  prior_filter_details='not inferred from history',
                                  preprocessing={'target_sfreq': target_sfreq, 'bandpass_applied': False,
                                                 'notch_applied': False, 'anti_alias_resampling': record.sfreq != target_sfreq},
                                  channel_harmonization=channel_policy)
                records.append(replace(record, signal_path=str(dest), channels=channels, sfreq=float(target_sfreq),
                                       quality={'before':before,'after':signal_quality(signal)}, provenance=provenance))
                Path(record.signal_path).unlink()
                if len(records) == 1:
                    signal_plot(signal, target_sfreq, processed/'diagnostic.png')
                if len(records) % 50 == 0:
                    print(f'{subject}: cached {len(records)} observations; source slot {int(record.sentence_id)+1}/{audit.get("total_sentences", "?")}', flush=True)
            if not records:
                raise ValueError('No finite sentence waveforms imported')
            validation = validate_records(records)
            if not validation['valid']:
                raise ValueError(f'Subject validation failed: {validation["errors"][:3]}')
            write_manifest(manifest, records)
            write_json(subject_dir/'validation.json', validation)
            write_json(subject_dir/'metadata.json', all_meta)
            write_json(state_path, {'signature':signature, 'manifest_digest':manifest_digest(records), 'audit':audit})
            combined.extend(records)
            report['subjects'][subject] = dict(audit, status='prepared', manifest=str(manifest))
            print(f'{subject}: prepared {len(records)}; skipped {len(audit["skipped"])} missing EEG slots', flush=True)
        except Exception as exc:
            report['subjects'][subject] = dict(audit, status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f'{subject}: FAILED: {exc}', flush=True)
        finally:
            gc.collect()
            write_json(report_path, report)
    failures = [s for s, info in report['subjects'].items() if info['status'] == 'failed']
    if not combined:
        report.update(status='failed', error='No subject was prepared')
        write_json(report_path, report)
        return report
    manifest = output / ('partial.jsonl' if failures else 'combined.jsonl')
    write_manifest(manifest, combined)
    validation = validate_records(combined)
    write_json(output/'validation.json', validation)
    report.update(status='partial' if failures else 'prepared', manifest=str(manifest),
                  n_observations=len(combined), n_subjects=len({r.subject for r in combined}),
                  n_unique_sentences=len({r.key for r in combined}), failed_subjects=failures,
                  subject_split_possible=len({r.subject for r in combined}) >= 3)
    if not validation['valid']:
        report.update(status='failed', error='Combined observation validation failed')
    if cache_text and not failures and validation['valid']:
        text_signature = {'keys':sorted({r.key for r in combined}), 'model':text_model, 'revision':text_revision}
        name = hashlib.sha256(json.dumps(text_signature, sort_keys=True).encode()).hexdigest()[:16]
        path = output / f'sentence_embeddings_{name}.npz'
        try:
            TextCache.build(combined, path, text_model, text_revision)
            report['text_cache'] = str(path)
        except Exception as exc:
            report.update(status='text_cache_failed', error=str(exc), data_ready=True)
    elif failures:
        report['text_cache_status'] = 'not_created_for_incomplete_cohort'
    write_json(report_path, report)
    return report
