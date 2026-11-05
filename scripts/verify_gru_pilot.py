"""Read-only checks of the completed pilot's saved evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from eeg_decoding.evaluation.retrieval import summarize

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/gru_pilot_v1_20261101'


def read(p): return json.loads(p.read_text())


def main():
    plan=read(RUN/'plan.json'); cfg=plan['configuration']; rows=read(RUN/'training_cv_all_scores.json')
    selected=read(RUN/'frozen_selection.json')['common_epoch']; summary=read(RUN/'training_cv_summary.json')
    expected={tuple(k) for k in plan['train_observations']}; pool=plan['training_candidate_keys']
    assert len(pool)==len(set(pool))==300 and len(expected)==2245
    query_union=set(); base=None
    for folder in sorted(RUN.glob('cv_*')):
        prov=read(folder/'provenance.json'); a={tuple(k) for k in prov['fit_observations']}; b={tuple(k) for k in prov['query_observations']}
        assert not a&b and a|b==expected and not query_union&b
        query_union |= b
        assert not set(prov['fit_readers'])&set(prov['query_readers'])
        assert prov['candidate_keys']==pool and prov['gaussian_exact_masks_match']
        assert all(k!=v for k,v in prov['class_shuffle_mapping'].items())
        state=torch.load(folder/'initial_weights.pt',map_location='cpu',weights_only=True)
        if base is None: base=state
        for key in base: torch.testing.assert_close(base[key],state[key],rtol=0,atol=0)
        scores=read(folder/'scores.json')
        assert len(scores)==len(cfg['conditions'])*len(cfg['epoch_candidates'])+1
        for score in scores:
            assert len(score['ranks'])==len(b)
            metric=summarize(score['ranks'])
            for m in ('recall@1','recall@5','recall@10','mrr'): assert abs(score[m]-metric[m])<1e-12
        for condition in cfg['conditions']:
            history=read(folder/condition/'history.json')
            assert [r['epoch'] for r in history]==list(range(1,13))
            assert all(np.isfinite(r['training_cosine_loss']) for r in history)
            for epoch in cfg['epoch_candidates']: assert (folder/condition/f'epoch_{epoch}.pt').is_file()
    assert query_union==expected
    for c,s in summary.items():
        selected_rows=[r for r in rows if r['condition']==c and r['epoch']==(0 if c=='structural_only' else selected)]
        assert len(selected_rows)==8
        for m in s['macro']: assert abs(s['macro'][m]-np.mean([r[m] for r in selected_rows]))<1e-12
    decision=read(RUN/'decision.json'); access=read(RUN/'access_summary.json')
    assert not access['test_arrays_loaded'] and not access['test_evaluated'] and not access['raw_source_files_opened']
    assert access['outer_evaluated']==decision['training_gate_passed']
    assert read(RUN/'preservation_after.json')['unchanged']
    result={'passed':True,'folds':8,'holdout_observations':len(query_union),'scored_rows':len(rows),
        'initial_weights_identical':True,'histories_complete':True,'saved_ranks_and_macro_metrics_verified':True,
        'reader_disjointness_verified':True,'no_test_access_in_declared_execution':True}
    (RUN/'evidence_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
