"""Source-validation regularization ablation using a completed local ridge cache.

Never edits the original run. Cache lineage and source split digests are saved;
target data is used only by the final checkpoint evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.retrieval import ranks, summarize
from eeg_decoding.models.classical import ridge_path
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.utils.config import load_config
from eeg_decoding.utils.logging import write_json
from eeg_decoding.workflow import evaluate_checkpoint, make_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--text-cache', required=True)
    parser.add_argument('--config', default='configs/transfer_ridge_regularized.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original, output = Path(args.original), Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh ablation output directory')
    meta = json.loads((original/'metadata.json').read_text())
    if meta.get('encoder_origin') or meta['stage'] != 'ridge':
        raise ValueError('This ablation requires statistical ridge feature caches')
    config = dict(meta['config'])
    config['ridge_alphas'] = load_config(args.config)['ridge_alphas']
    records = read_manifest(args.manifest)
    protocol = meta['splits']['protocol']
    source, target = meta['train_datasets'][0], meta['test_datasets'][0]
    splits = make_splits(records, config, protocol, source, target)
    for name in ['train', 'val']:
        if manifest_digest(splits[name]) != meta['splits']['splits'][name]['digest']:
            raise ValueError(f'{name} data changed since original feature computation')
    text = TextCache(args.text_cache)
    if text.digest != meta['text_cache_digest']:
        raise ValueError('Text cache changed')
    train = np.load(original/'train_features.npy', mmap_mode='r')
    val = np.load(original/'val_features.npy', mmap_mode='r')
    if train.shape[0] != len(splits['train']) or val.shape[0] != len(splits['val']):
        raise ValueError('Feature rows do not match source splits')
    keys = [r.key for r in splits['train']]
    if meta['training_control'] == 'shuffled':
        keys = shuffled_keys(keys, config.get('seed', 42))
    pool = sorted({r.key for r in splits['val']})
    best, chosen, selected, table = -float('inf'), None, None, []
    start = time.perf_counter()
    for alpha, weights in ridge_path(train, text.get(keys), config['ridge_alphas']):
        prediction = ((val-weights['mean'])/weights['scale']) @ weights['coef'].T + weights['intercept']
        metric = summarize(ranks(prediction, text.get(pool), [r.key for r in splits['val']], pool))['mrr']
        table.append(dict(alpha=alpha, validation_mrr=metric))
        if metric > best:
            best, chosen, selected = metric, weights, alpha
    fit_seconds = time.perf_counter()-start
    updated = copy.deepcopy(meta)
    updated.update(config=config, selection=table, selected_alpha=selected,
                   ridge_path_implementation='shared_train_svd',
                   ablation='extended_source_validation_regularization',
                   original_run=str(original.resolve()), fit_and_selection_seconds=fit_seconds,
                   feature_cache_sha256={name: hashlib.sha256((original/name).read_bytes()).hexdigest()
                                         for name in ['train_features.npy', 'val_features.npy']})
    output.mkdir(parents=True)
    np.savez_compressed(output/'ridge.npz', **chosen)
    write_json(output/'metadata.json', updated)
    write_json(output/'splits.json', updated['splits'])
    print(f'Selected alpha={selected}; source validation MRR={best:.6f}; fit/selection={fit_seconds:.2f}s', flush=True)
    report = evaluate_checkpoint(records, output/'ridge.npz', args.text_cache, output/'evaluation.json')
    print(json.dumps(report['metrics']), flush=True)


if __name__ == '__main__':
    main()
