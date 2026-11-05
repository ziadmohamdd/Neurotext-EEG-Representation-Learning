"""Six fixed fits; validation retrieval only. Never imports the test evaluator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.evaluation.retrieval import ranks, summarize, chance
from eeg_decoding.models.temporal_pooling import matched_pooling_models, initialization_report, state_hash
from eeg_decoding.training.alignment_train import TextCache, train_alignment
from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.utils.config import load_config
from eeg_decoding.utils.device import device_info
from eeg_decoding.utils.logging import write_json
from eeg_decoding.utils.reproducibility import seed_all
from eeg_decoding.workflow import make_splits, prepare_metadata, loader

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_gate(reports):
    """Fixed descriptive screen, not a significance test or permission to test."""
    by = {(r['pooling'], r['control']): r for r in reports}
    checks = {}
    g, f = by['global', 'real'], by['four_segment', 'real']
    checks['real_mrr_improves'] = f['metrics']['mrr'] > g['metrics']['mrr']
    for control in ('gaussian', 'shuffled'):
        gc, fc = by['global', control], by['four_segment', control]
        checks[f'real_beats_{control}_mrr'] = f['metrics']['mrr'] > fc['metrics']['mrr']
        checks[f'mrr_gap_improves_vs_{control}'] = (f['metrics']['mrr'] - fc['metrics']['mrr'] > g['metrics']['mrr'] - gc['metrics']['mrr'] + 1e-12)
        for subject in f['per_subject']:
            checks[f'{subject}_real_beats_{control}_mrr'] = f['per_subject'][subject]['mrr'] > fc['per_subject'][subject]['mrr']
    checks['recall_corroboration'] = sum(
        f['metrics'][k] > max(by['four_segment', c]['metrics'][k] for c in ('gaussian', 'shuffled'))
        and f['metrics'][k] >= g['metrics'][k] for k in ('recall@1', 'recall@5', 'recall@10')) >= 2
    return {'checks': checks, 'useful_validation_improvement': all(checks.values()),
            'test_evaluated': False, 'interpretation': 'Descriptive screen on one previously used validation split; no independent-cohort or significance claim.'}


@torch.no_grad()
def validation_report(model, rows, meta, text, device, pooling, control):
    expected = meta['splits']['splits']['val']
    if manifest_digest(rows) != expected['digest']:
        raise ValueError('Only the saved validation rows may be evaluated')
    if set(expected['subjects']) & set(meta['splits']['splits']['test']['subjects']):
        raise ValueError('Validation/test overlap')
    from eeg_decoding.preprocessing.normalization import ChannelNormalizer
    norm = ChannelNormalizer.from_state(meta['normalizer'])
    model.to(device).eval()
    predicted = []
    for batch in loader(rows, meta['config'], norm, control='gaussian' if control == 'gaussian' else 'real'):
        predicted.append(model(batch['eeg'].to(device), batch['mask'].to(device)).cpu().numpy())
    pool = sorted({r.key for r in rows})
    rank = ranks(np.concatenate(predicted), text.get(pool), [r.key for r in rows], pool)
    report = {'pooling': pooling, 'control': control, 'evaluation_split': 'val',
              'metrics': summarize(rank), 'chance': chance(len(pool)), 'candidate_keys': pool,
              'candidate_pool_sha256': hashlib.sha256(json.dumps(pool).encode()).hexdigest(),
              'validation_manifest_digest': manifest_digest(rows), 'per_subject': {},
              'queries': [{'identity': [r.dataset,r.subject,r.session,r.task,r.sentence_id], 'rank': float(v)} for r,v in zip(rows,rank)]}
    for subject in sorted({r.subject_key for r in rows}):
        report['per_subject'][subject] = summarize(rank[[r.subject_key == subject for r in rows]])
    return report


def main(output):
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    seed_all(42)
    config = load_config(ROOT / 'configs/temporal_pooling_comparison.yaml')
    old = ROOT / 'results/cnn_candidate_comparison_20261030'
    prior = json.loads((old/'plan.json').read_text())
    manifest = Path(prior['manifest']); text_path = Path(prior['text_cache'])
    if sha(manifest) != prior['manifest_sha256'] or sha(text_path) != prior['text_cache_sha256']:
        raise ValueError('Original manifest/text cache changed; punctuation repair forbidden')
    for name, value in prior['common_config'].items():
        if config[name] != value:
            raise ValueError(f'Changed fixed setting: {name}')
    if config['alignment_candidate_scope'] != 'batch':
        raise ValueError('Keep original validation-selected batch loss fixed')
    records = read_manifest(manifest); text = TextCache(text_path)
    splits = make_splits(records, config)
    norm, meta = prepare_metadata(records, splits, config, 'alignment', 'subject')
    saved = json.loads((old/'batch_real/splits.json').read_text())
    if meta['splits'] != saved:
        raise ValueError('Split differs from existing experiment')
    old_meta = load_checkpoint(old/'batch_real/best.pt')
    if norm.state_dict() != old_meta['normalizer']:
        raise ValueError('Normalization changed')
    baseline, variant = matched_pooling_models(len(meta['channels']), config['embedding_dim'], text.embeddings.shape[1], 42, config['quality_aware'])
    init = initialization_report(baseline, variant)
    if init['global_initial_sha256'] != old_meta['initial_model_sha256']:
        raise ValueError('Baseline initialization differs from existing CNN')
    protected = [p for p in (ROOT/'results').rglob('*') if p.is_file()] + [manifest, text_path, ROOT/'data/processed/zuco1_batch/combined.jsonl']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    output.mkdir(parents=True)
    write_json(output/'preservation_before.json', before)
    device, info = device_info(config['device'])
    meta.update(text_cache_digest=text.digest, encoder_origin='supervised_cnn', device_info=info)
    jobs = [{'pooling': pool, 'control': c, 'status': 'planned'} for pool in ('global','four_segment') for c in ('real','gaussian','shuffled')]
    write_json(output/'plan.json', {'common_config':config, 'loss':'unchanged symmetric batch multi-positive CE, temperature .07',
        'manifest':str(manifest), 'manifest_sha256':sha(manifest), 'text_cache':str(text_path), 'text_cache_sha256':text.digest,
        'saved_split_source':str(old/'batch_real/splits.json'), 'split_digest':meta['split_digest'], 'splits':meta['splits'],
        'normalizer':norm.state_dict(), 'initialization':init, 'device':info,
        'training_candidate_keys':sorted({r.key for r in splits['train']}), 'validation_candidate_keys':sorted({r.key for r in splits['val']}),
        'pooling_definition':'AdaptiveAvgPool1d(1) versus AdaptiveAvgPool1d(4), channel-major concatenate then linear head. Non-divisible lengths use PyTorch overlapping boundary bins.',
        'budget':'Same maximum 50 epochs, patience 8, validation batch CE best-epoch selection. Actual epochs may differ.',
        'validation_screen':'Higher real MRR; positive and improved real-minus-both-control MRR gaps; positive four-segment MRR gaps for each validation subject; >=2 recall cutoffs above both controls and no worse than global real.',
        'test_policy':'Validation-only in this execution even if screen passes. Never call test evaluation. Stop on no useful control-relative improvement.',
        'punctuation_repair_applied':False, 'one_seed':42})
    for source in ['src/eeg_decoding/models/eeg_cnn.py','src/eeg_decoding/models/temporal_pooling.py',
                   'src/eeg_decoding/models/projection.py','src/eeg_decoding/training/alignment_train.py',
                   'src/eeg_decoding/training/trainer.py','src/eeg_decoding/workflow.py','configs/local.yaml',
                   'configs/cnn_candidate_comparison.yaml','configs/temporal_pooling_comparison.yaml',
                   'tests/test_temporal_pooling.py',__file__]:
        dest = output/'implementation'/Path(source).name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(source,dest)
    reports = []
    for job in jobs:
        folder = output/(job['pooling']+'_'+job['control']); folder.mkdir()
        seed_all(42)
        pair = matched_pooling_models(len(meta['channels']), config['embedding_dim'], text.embeddings.shape[1],42,config['quality_aware'])
        model = pair[0 if job['pooling']=='global' else 1]
        cfg = dict(config, temporal_segments=1 if job['pooling']=='global' else 4)
        metadata = dict(meta, config=cfg, training_control=job['control'], temporal_pooling_initialization=init)
        write_json(folder/'config.json', cfg); write_json(folder/'splits.json',meta['splits'])
        job['status']='training'; write_json(output/'jobs.json',jobs)
        print(f"TRAIN {job['pooling']}/{job['control']} validation-only",flush=True)
        started = time.monotonic()
        history = train_alignment(model,loader(splits['train'],cfg,norm,True,True,job['control']),
                    loader(splits['val'],cfg,norm,False,True,'gaussian' if job['control']=='gaussian' else 'real'),
                    text,cfg,device,folder,metadata)
        best = load_checkpoint(folder/'best.pt',model)
        job.update(status='trained',best_epoch=best['epoch'],epochs_run=len(history),validation_loss=best['best'],
                   checkpoint_sha256=sha(folder/'best.pt'),seconds=time.monotonic()-started)
        assert best['initial_model_sha256'] == init['global_initial_sha256' if job['pooling']=='global' else 'four_segment_initial_sha256']
        write_json(output/'jobs.json',jobs)
        # Checkpoint selection is already complete; retrieval never selects epochs.
        report = validation_report(model,splits['val'],best,text,device,job['pooling'],job['control'])
        report.update(best_epoch=best['epoch'],validation_loss=best['best'],epochs_run=len(history))
        write_json(folder/'validation.json',report); reports.append(report)
        job['status']='validation_evaluated';write_json(output/'jobs.json',jobs)
        print(json.dumps({'pooling':job['pooling'],'control':job['control'],'epochs':len(history),'best_epoch':best['epoch'],'validation':report['metrics']}),flush=True)
        del model, pair
        if device.type=='cuda':torch.cuda.empty_cache()
    write_json(output/'validation_summary.json',reports)
    decision = validation_gate(reports); write_json(output/'decision.json',decision)
    gaps = []
    lookup = {(r['pooling'],r['control']):r for r in reports}
    for pooling in ('global','four_segment'):
        for control in ('gaussian','shuffled'):
            gaps.append({'pooling':pooling,'comparison':'real_minus_'+control,
                         **{k:lookup[pooling,'real']['metrics'][k]-lookup[pooling,control]['metrics'][k] for k in ('recall@1','recall@5','recall@10','mrr')}})
    write_json(output/'control_gaps.json',gaps)
    rows = [dict(pooling=r['pooling'],control=r['control'],best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],validation_loss=r['validation_loss'],**r['metrics']) for r in reports]
    with (output/'validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    after = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    if after != before:raise RuntimeError('A previous artifact changed')
    write_json(output/'preservation_after.json',{'unchanged':True,'protected_files':len(before),'files_sha256':after})
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
