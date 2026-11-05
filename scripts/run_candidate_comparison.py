"""One preregistered within-ZuCo1 CNN comparison; inspect test only after all fits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.data.validation import manifest_digest
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.utils.config import load_config
from eeg_decoding.utils.logging import write_json
from eeg_decoding.workflow import make_splits,train_neural,evaluate_checkpoint


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--audit',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a fresh comparison output')
    audit=json.loads(Path(args.audit).read_text())
    if audit['status']!='passed' or audit['cohort_observations']!=3349:
        raise ValueError('Review source-pairing audit before training the fixed cohort')
    manifest=Path('data/manifests/zuco1_duration_qc.jsonl')
    text_path=Path(json.loads(Path('data/processed/zuco1_batch/summary.json').read_text())['text_cache'])
    config=load_config('configs/cnn_candidate_comparison.yaml')
    records=read_manifest(manifest)
    splits=make_splits(records,config)
    text=TextCache(text_path)
    jobs=[dict(scope=scope,control=control,status='not_executed')
          for scope in ['batch','training_unique'] for control in ['real','gaussian','shuffled']]
    output.mkdir(parents=True)
    write_json(output/'plan.json',dict(dataset='zuco1',protocol='subject',seed=42,
        audit=str(Path(args.audit).resolve()),audit_sha256=sha(args.audit),
        manifest=str(manifest.resolve()),manifest_sha256=sha(manifest),text_cache=str(text_path.resolve()),
        text_cache_sha256=text.digest,text_metadata=text.metadata,common_config=config,
        training_candidate_keys=sorted({r.key for r in splits['train']}),
        validation_candidate_keys=sorted({r.key for r in splits['val']}),
        test_candidate_keys=sorted({r.key for r in splits['test']}),
        split_digests={name:manifest_digest(rows) for name,rows in splits.items()},
        subjects={name:sorted({r.subject_key for r in rows}) for name,rows in splits.items()},
        selection='All settings fixed to the existing local recipe; best epoch and early stopping use common validation batch CE only',
        loss_change='Only forward EEG-to-text denominator expands to unique training sentences; reverse batch term and .5 weights unchanged',
        test_policy='Train all six jobs; freeze chosen checkpoint hashes; then evaluate test once per chosen artifact',
        limitations='One exploratory subject split/seed, previously inspected cohort; no generalization success from training or isolated test gains'))
    shutil.copy2(args.audit,output/'source_audit.json')
    for source in ['configs/local.yaml','configs/cnn_candidate_comparison.yaml',
                   'src/eeg_decoding/models/projection.py','src/eeg_decoding/training/alignment_train.py',
                   'src/eeg_decoding/training/trainer.py','src/eeg_decoding/models/eeg_cnn.py',
                   'src/eeg_decoding/workflow.py',__file__]:
        dest=output/'implementation'/Path(source).name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest)
    write_json(output/'jobs.json',jobs)
    for job in jobs:
        folder=output/(job['scope']+'_'+job['control'])
        cfg=dict(config,alignment_candidate_scope=job['scope'])
        print(f"Training {job['scope']}/{job['control']} (validation-only checkpoint selection)",flush=True)
        job['status']='training';write_json(output/'jobs.json',jobs)
        train_neural(records,cfg,'alignment',folder,text_path,control=job['control'])
        meta=load_checkpoint(folder/'best.pt')
        job.update(status='trained',best_epoch=meta['epoch'],validation_loss=meta['best'],
                   best_checkpoint_sha256=sha(folder/'best.pt'),
                   initial_model_sha256=meta['initial_model_sha256'])
        write_json(output/'jobs.json',jobs)
        print(f"Selected epoch {meta['epoch']}; common validation loss={meta['best']:.6f}",flush=True)
    metas=[load_checkpoint(output/(j['scope']+'_'+j['control'])/'best.pt') for j in jobs]
    for meta in metas:
        assert meta['split_digest']==metas[0]['split_digest']
        assert meta['normalizer']==metas[0]['normalizer']
        assert meta['initial_model_sha256']==metas[0]['initial_model_sha256']
        assert meta['candidate_keys']==metas[0]['candidate_keys']
        assert meta['channels']==metas[0]['channels'] and meta['sfreq']==metas[0]['sfreq']
    write_json(output/'selection_frozen.json',dict(jobs=jobs,test_evaluation_started=False,
               model_selection='common validation batch CE; no test scores inspected in this experiment before freezing'))
    rows=[]
    for job,meta in zip(jobs,metas):
        folder=output/(job['scope']+'_'+job['control'])
        assert sha(folder/'best.pt')==job['best_checkpoint_sha256']
        print(f"Evaluating frozen {job['scope']}/{job['control']}",flush=True)
        report=evaluate_checkpoint(records,folder/'best.pt',text_path,folder/'evaluation.json')
        report['alignment_candidate_scope']=job['scope']
        report['comparison']='fixed_zuco1_candidate_denominator_seed42'
        write_json(folder/'evaluation.json',report)
        rows.append(dict(scope=job['scope'],control=job['control'],best_epoch=meta['epoch'],
                         validation_loss=meta['best'],**report['metrics'],
                         n_candidates=len(report['candidate_keys']),candidate_pool_sha256=report['candidate_pool_sha256']))
        job['status']='executed'; write_json(output/'jobs.json',jobs)
    write_json(output/'summary.json',rows)
    with (output/'retrieval.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__': main()
