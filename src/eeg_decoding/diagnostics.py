"""Tiny training-set memorization check; never a held-out decoding result."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .data.dataset import read_manifest, write_manifest, load_signal
from .data.validation import manifest_digest, validate_records
from .evaluation.cross_subject import group_split
from .evaluation.retrieval import ranks, summarize, chance
from .models.eeg_cnn import EEGCNN
from .models.projection import EEGTextModel, alignment_loss
from .preprocessing.normalization import ChannelNormalizer
from .training.alignment_train import TextCache
from .training.checkpoints import save_checkpoint
from .training.trainer import Trainer
from .utils.config import load_config
from .utils.device import device_info
from .utils.logging import write_json
from .utils.reproducibility import seed_all
from .workflow import loader


def select_tiny_training_set(records, seed, n_sentences):
    if not 2 <= n_sentences <= 16:
        raise ValueError('Choose 2 to 16 distinct training sentences')
    if len({r.dataset for r in records}) != 1:
        raise ValueError('Run the diagnostic on one dataset version at a time')
    splits = group_split(records, seed)
    for subject in sorted({r.subject_key for r in splits['train']}):
        unique = {}
        for r in sorted(splits['train'], key=lambda r: (r.key, r.session, r.sentence_id)):
            if r.subject_key == subject:
                unique.setdefault(r.key, r)
        if len(unique) >= n_sentences:
            return list(unique.values())[:n_sentences], splits
    raise ValueError('No training subject has enough distinct sentences')


def overfit_diagnostic(records, text_path, config, output, steps=200, n_sentences=8):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh diagnostic output directory')
    if steps < 1:
        raise ValueError('steps must be positive')
    config = dict(config, batch_size=n_sentences, gradient_accumulation_steps=1,
                  mixed_precision='disabled', weight_decay=0., num_workers=0)
    seed = config.get('seed', 42)
    seed_all(seed)
    selected, splits = select_tiny_training_set(records, seed, n_sentences)
    validation = validate_records(selected)
    if not validation['valid']:
        raise ValueError(f'Diagnostic signal validation failed: {validation["errors"]}')
    if any(r.channels != selected[0].channels or r.sfreq != selected[0].sfreq for r in selected):
        raise ValueError('Diagnostic signals must have matching channels and sampling rate')
    if any(r.provenance.get('units') != 'uV' for r in selected):
        raise ValueError('Diagnostic requires documented physical microvolt units')
    cache = TextCache(text_path)
    keys = [r.key for r in selected]
    targets = cache.get(keys)
    cosine = targets @ targets.T
    off_diagonal = cosine[~np.eye(n_sentences, dtype=bool)]
    signal_audit = []
    signals = []
    for r in selected:
        x = load_signal(r)
        signal_audit.append(dict(sentence_key=r.key, text=r.text, subject=r.subject_key,
                                 sentence_id=r.sentence_id, n_samples=x.shape[1],
                                 duration_seconds=x.shape[1]/r.sfreq,
                                 flat_channels=int((x.std(axis=1)<1e-6).sum()),
                                 maximum_absolute_uv=float(np.abs(x).max())))
        signals.append(x)
    duplicates = [[keys[i], keys[j]] for i in range(len(signals)) for j in range(i)
                  if np.array_equal(signals[i], signals[j])]
    norm = ChannelNormalizer().fit(selected, {r.subject_key for r in selected})
    device, info = device_info(config.get('device', 'auto'))
    # Keep one fixed full batch: every diagnostic target is a contrastive candidate.
    batch = next(iter(loader(selected, config, norm, alignment=True)))
    batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    target_tensor = torch.from_numpy(cache.get(batch['keys'])).to(device)
    encoder = EEGCNN(len(selected[0].channels), config.get('embedding_dim',128),
                     quality_aware=config.get('quality_aware',False))
    model = EEGTextModel(encoder, config.get('embedding_dim',128), targets.shape[1])
    output.mkdir(parents=True)
    trainer = Trainer(model, config, device, output)

    def objective(m, b):
        return alignment_loss(m(b['eeg'], b['mask']), target_tensor, b['keys'])

    audit = dict(scope='Training subset only; source annotation timing is not independently verified',
                 records=signal_audit, identical_signal_pairs=duplicates,
                 target_pair_cosine_min=float(off_diagonal.min()),
                 target_pair_cosine_max=float(off_diagonal.max()),
                 target_pair_cosine_mean=float(off_diagonal.mean()),
                 held_out_subjects={name:sorted({r.subject_key for r in splits[name]}) for name in ['val','test']})
    write_json(output/'audit.json',audit)
    write_manifest(output/'selected.jsonl',selected)
    history=[]

    def measure(step):
        model.eval()
        with torch.no_grad():
            predicted = model(batch['eeg'],batch['mask'])
            loss = float(alignment_loss(predicted,target_tensor,batch['keys']))
        metrics = summarize(ranks(predicted.cpu().numpy(),targets,batch['keys'],keys))
        history.append(dict(step=step,loss=loss,training_metrics=metrics))
        write_json(output/'history.json',history)
        print(f'Step {step}/{steps}: loss={loss:.6f}, training R@1={metrics["recall@1"]:.1%}, '
              f'MRR={metrics["mrr"]:.4f}',flush=True)
        return metrics,loss

    measure(0)
    for step in range(1,steps+1):
        trainer.epoch([batch],objective,True)
        if step % 10 == 0 or step == steps:
            metrics,loss = measure(step)
            if metrics['recall@1'] == 1. and loss < .05:
                break
    report = dict(status='diagnostic_completed',protocol='training_subset_memorization_only',
                  generalization_evidence=False, config=config, device_info=info,
                  n_sentences=n_sentences,steps_completed=history[-1]['step'],
                  source_training_subset_digest=manifest_digest(selected),
                  text_cache_digest=cache.digest,normalizer=norm.state_dict(),
                  initial=history[0],final=history[-1],chance=chance(n_sentences),
                  memorization_achieved=history[-1]['training_metrics']['recall@1']==1.,
                  interpretation='Memorization can test optimization capacity; it does not verify EEG/text timing or held-out accuracy',
                  synthetic=any(r.provenance.get('synthetic') for r in selected))
    save_checkpoint(output/'diagnostic.pt',model,stage='diagnostic_overfit',diagnostic=report)
    write_json(output/'diagnostic.json',report)
    print(f'Diagnostic: {output/"diagnostic.json"}; not an evaluation for export',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--text-cache',required=True)
    parser.add_argument('--config',default='configs/local.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--sentences',type=int,default=8)
    args=parser.parse_args()
    overfit_diagnostic(read_manifest(args.manifest),args.text_cache,load_config(args.config),
                       args.output,args.steps,args.sentences)


if __name__=='__main__':
    main()
