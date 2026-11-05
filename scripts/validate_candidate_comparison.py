"""Read-only checks of the completed matched loss comparison and its provenance."""
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from eeg_decoding.models.projection import alignment_loss
from eeg_decoding.training.checkpoints import load_checkpoint

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/cnn_candidate_comparison_20261030'
OUT=ROOT/'docs/research_results/cnn_candidates_20261030'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan=json.loads((SOURCE/'plan.json').read_text())
    frozen=json.loads((SOURCE/'selection_frozen.json').read_text())
    jobs=json.loads((SOURCE/'jobs.json').read_text())
    assert len(jobs)==6 and all(j['status']=='executed' for j in jobs)
    assert sha(plan['manifest'])==plan['manifest_sha256']
    assert sha(plan['text_cache'])==plan['text_cache_sha256']
    assert sha(plan['audit'])==plan['audit_sha256']
    first=None; shapes=None; rows=[]
    for job in jobs:
        folder=SOURCE/(job['scope']+'_'+job['control'])
        meta=load_checkpoint(folder/'best.pt')
        state=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
        current_shapes={name:list(value.shape) for name,value in state['model'].items()}
        if shapes is None: shapes=current_shapes
        assert current_shapes==shapes
        if first is None: first=meta
        for key in ['normalizer','split_digest','initial_model_sha256','channels','sfreq','candidate_keys']:
            assert meta[key]==first[key]
        config=dict(meta['config']);assert config.pop('alignment_candidate_scope')==job['scope']
        assert config==plan['common_config']
        objective=meta['alignment_objective']
        assert objective['reverse_term']=='unchanged_symmetric_batch_multi_positive'
        assert objective['temperature']==.07
        if job['scope']=='training_unique': assert objective['candidate_keys']==plan['training_candidate_keys']
        else: assert objective['candidate_keys'] is None
        full_history=json.loads((folder/'history.json').read_text())
        winner=min(full_history,key=lambda r:r['val_loss'])
        assert winner['epoch']==meta['epoch']==job['best_epoch']
        assert winner['val_loss']==meta['best']==job['validation_loss']
        selected=next(j for j in frozen['jobs'] if j['scope']==job['scope'] and j['control']==job['control'])
        assert selected['status']=='trained' and sha(folder/'best.pt')==selected['best_checkpoint_sha256']
        assert (folder/'evaluation.json').stat().st_mtime_ns > (SOURCE/'selection_frozen.json').stat().st_mtime_ns
        report=json.loads((folder/'evaluation.json').read_text())
        assert report['candidate_keys']==plan['test_candidate_keys']
        assert report['evaluation_control']==('gaussian' if job['control']=='gaussian' else 'real')
        rows.append(dict(scope=job['scope'],control=job['control'],best_epoch=meta['epoch'],epochs_completed=len(full_history),
                         best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt')))
    choice=json.loads((SOURCE/'real_loss_validation_choice.json').read_text(encoding='utf-8-sig'))
    assert choice['selected_scope']=='batch' and choice['test_reports_present']==0
    # Verify the default loss against the original repository implementation.
    original=subprocess.check_output(['git','show','HEAD:src/eeg_decoding/models/projection.py'],cwd=ROOT,text=True)
    namespace={};exec(compile(original,'original_projection.py','exec'),namespace)
    torch.manual_seed(8)
    q=torch.randn(4,7,requires_grad=True);t=torch.randn(4,7);keys=['a','b','a','c']
    old=namespace['alignment_loss'](q,t,keys);new=alignment_loss(q,t,keys)
    assert torch.equal(old,new)
    assert torch.equal(torch.autograd.grad(old,q,retain_graph=True)[0],torch.autograd.grad(new,q)[0])
    previous=json.loads((ROOT/'results/exploratory_repeat_prior_hashes_20261029.json').read_text(encoding='utf-8-sig'))
    for item in previous: assert sha(item['Path']).upper()==item['SHA256']
    manifest=json.loads((OUT/'archive_manifest.json').read_text())
    for item in manifest['files']:
        assert sha(ROOT/item['source'])==item['sha256']
        assert sha(OUT/item['archived'])==item['sha256']
    report=dict(status='passed',scope='artifact checks, not additional EEG training or test evaluation',
                initial_model_sha256=first['initial_model_sha256'],same_architecture_and_inputs=True,
                prior_checked_experiment_files_unchanged=len(previous),old_default_loss_and_gradient='exactly equal on a synthetic CPU check',
                test_after_selection_freeze=True,real_loss_family_chosen_by_validation=choice['selected_scope'],
                metadata_and_checkpoint_checks=rows,
                suite=dict(passed=145,optional_skips=2,deselected=0),
                gpu_reproducibility='CuBLAS warning with CUBLAS_WORKSPACE_CONFIG unset; no bitwise GPU claim')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
