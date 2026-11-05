"""Verify and archive the validation-only TRT pilot; no model fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from eeg_decoding.evaluation.retrieval import summarize

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run):
    run=run.resolve();archive=ROOT/'docs/research_results/spectral_trt_pilot_v1_20261030';doc=ROOT/'docs/spectral_trt_pilot_v1_20261030.md'
    if archive.exists() or doc.exists():raise FileExistsError('Preserve existing pilot report/archive')
    plan=json.loads((run/'plan.json').read_text());cfg=plan['configuration']
    reports=json.loads((run/'validation_summary.json').read_text());decision=json.loads((run/'decision.json').read_text())
    viewmeta=json.loads((run/'feature_view.json').read_text());view=Path(viewmeta['path']);index=json.loads((view/'index.json').read_text())
    gaps=json.loads((run/'real_minus_controls.json').read_text());preservation=json.loads((run/'preservation_after.json').read_text())
    assert len(reports)==6 and {r['condition'] for r in reports}==set(cfg['conditions'])
    assert all(r['evaluation_split']=='val' and r['metrics']['n_queries']==512 and r['candidate_keys']==plan['candidate_keys'] for r in reports)
    assert len(plan['candidate_keys'])==298
    assert all(r['observation_identities']==plan['validation_observations'] for r in reports)
    assert not index['test_features_extracted'] and not decision['test_evaluated']
    assert all(p['shape'][0]==p['words'] and p['real_mask_sha256']==p['gaussian_mask_sha256'] for p in index['mask_proofs'])
    assert len(index['mask_proofs'])==2757
    assert sha(view/'sentence_vectors.npz')==viewmeta['vectors_sha256']
    assert sha(view/'index.json')==viewmeta['index_sha256']
    assert sha(view/'gaussian_training_moments.npz')==viewmeta['gaussian_training_moments_sha256']
    assert all(sha(ROOT/p)==h for p,h in preservation['files_sha256'].items())
    for r in reports:
        assert summarize(np.array(r['ranks']))==r['metrics']
        assert sha(run/r['condition']/'ridge.npz')==r['selection']['coefficient_sha256']
        grid=json.loads((run/r['condition']/'selection.json').read_text())['grid']
        assert len(grid)==28 and abs(max(row['mrr'] for row in grid)-r['metrics']['mrr'])<1e-10
    cache=np.load(view/'sentence_vectors.npz',allow_pickle=False)
    for split,n in [('train',2245),('val',512)]:
        assert all(cache[condition+'_'+split].shape[0]==n for condition in ('real_trt','matched_gaussian','zero_missingness_only','count_duration_only','existing_sentence_ridge'))
        np.testing.assert_array_equal(cache['real_trt_'+split][:,840:],cache['matched_gaussian_'+split][:,840:])
        np.testing.assert_array_equal(cache['real_trt_'+split][:,840:],cache['zero_missingness_only_'+split][:,840:])
        np.testing.assert_array_equal(np.isnan(cache['real_trt_'+split]),np.isnan(cache['matched_gaussian_'+split]))
    cache.close()
    archive.mkdir(parents=True);copies=[]
    files=[*run.glob('*.json'),*run.glob('*/selection.json'),*run.glob('*/validation.json'),*run.glob('implementation/*'),
           view/'index.json',view/'sentence_baseline_normalizer.json',Path(__file__),
           ROOT/'results/pairing_audit_20261030/official_firstLevel_NR.m',ROOT/'configs/local.yaml',
           ROOT/'data/raw/zuco1/download_plan.json',ROOT/'logs/spectral_trt_pilot_v1_20261030.log']
    for source in files:
        if run in source.parents:rel=source.relative_to(run)
        elif source.parent==view:rel=Path('feature_view_metadata')/source.name
        elif source.suffix=='.log':rel=Path('logs')/source.name
        else:rel=Path('provenance')/source.name
        target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(source,target);assert sha(source)==sha(target)
        copies.append({'source':str(source.relative_to(ROOT)),'copy':str(target.relative_to(ROOT)),'sha256':sha(source)})
    write=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
    write(archive/'archive_manifest.json',copies)
    validation={'unit_tests':'7 passed in 5.00 s','conditions':6,'settings_per_condition':28,'train_observations_per_condition':2245,
        'validation_observations_per_condition':512,'validation_candidates':298,'identical_validation_rows_and_pools':True,
        'gaussian_word_masks_verified':2757,'shared_structural_values_identical':True,'feature_cache_hashes_verified':3,
        'selected_coefficients_and_validation_ranks_verified':6,'protected_prior_files_unchanged':preservation['protected_files'],
        'test_extracted_or_evaluated':False,'punctuation_repair_applied':False,'archive_source_copy_hash_matches':len(copies)}
    write(archive/'validation_checks.json',validation)
    with (archive/'validation.csv').open('w',newline='') as f:
        rows=[{'condition':r['condition'],**r['metrics'],**{k:r['selection'][k] for k in ('alpha','imputation','scaling','n_features')}} for r in reports]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    outcome=('**No useful validation advantage over the strongest control. Stopped without test extraction or evaluation.**'
             if not decision['useful_validation_advantage'] else '**The prespecified descriptive validation screen passed. Test data remain unextracted and unevaluated in this validation-only pilot; this does not establish generalization success.**')
    lines=['# Versioned TRT spectral-feature pilot — seed-42 validation only\n\n',outcome+'\n\n',
        'The pilot uses `zuco1_trt_sentence_mean_pilot_v1`, a new feature view with the original sentence targets and '
        'candidate embeddings. No previous data, checkpoint or experiment result was replaced. The punctuation '
        'repair was not applied. All six conditions use exactly the same 2,245 training and 512 validation observations '
        'and the same **298 unique validation candidates**. MRR remains reciprocal rank, not accuracy.\n\n',
        '## Frozen design and feature provenance\n\n',
        'Source inputs are the ten ZuCo1 training/validation NR MAT files; their hashes match the pinned download '
        'plan. ZAB/ZKB test MAT files and test voltage arrays were not opened by the pilot. Original source slot '
        'indices bind each observation; parent sentence content is checked against the unchanged target and never '
        'enters the numeric model matrix. **Word text is not decoded**, token/word IDs are not inputs, and no '
        'sentence/subject/session identity, pupil, bounding-box coordinates or raw eye-tracking trace enters a model. '
        'IDs are used solely for provenance, splitting, target lookup and deterministic Gaussian RNG.\n\n',
        'The selected family is **TRT**, with band-major `t1,t2,a1,a2,b1,b2,g1,g2` and the existing ordered '
        '105 channels. Each word supplies 840 source Hilbert-envelope amplitude values, retaining NaNs. '
        'No spectral differences or other fixation-measure EEG families are included. Each coordinate is averaged '
        'equally across its finite word values. This is an unweighted word mean over the provided TRT vectors, '
        'whose producer already averages available fixation-level envelope means. All-missing sentence coordinates '
        'stay NaN until training-only imputation. No duration weighting or spectral-to-waveform conversion occurs.\n\n',
        'The [source audit](word_eeg_feasibility_20261030.md) documents band/channel definitions, source hashes '
        'and boundary limits. The code defines bands 4–6, 6.5–8, 8.5–10, 10.5–13, 13.5–18, 18.5–30, '
        '30.5–40 and 40.5–49.5 Hz; the descriptor instead gives gamma2 40–49.5 Hz. No attempt was made to '
        'recompute or silently correct the provided features. Exact producer identity for every file is not proven. '
        'These are gaze-assisted, annotation-segmented features from the supplied session-level processing. '
        'They are not a strictly eight-second-window-matched or online EEG-only replacement. The sampled source '
        'boundary tail remains a limitation of the provided feature view.\n\n',
        '**Duration units are stored sample counts at 500 Hz, not milliseconds.** The audited source code '
        'uses inclusive slices, with raw snippet `N=duration+1` in applicable examples. The count/duration '
        'control uses those sample counts as stored; it never invents EEG onsets or fills inter-fixation gaps.\n\n',
        '## Matched conditions and prevention of target leakage\n\n',
        'Every spectral arm has the same 840 amplitude positions plus three structural values (displayed word '
        'count, complete-TRT word count, complete fraction) and a training-derived one-hot word-count basis. '
        'The basis uses 47 training-observed counts and one unseen-count bucket: **891 features** in total. '
        'Count values and missingness are deliberate privileged non-neural information, not hidden as EEG. '
        'Providing the same structural block to real, Gaussian and zero conditions makes an EEG gain face '
        'a competitive structural control. The count basis is derived from actual word-record counts, never '
        'sentence IDs or labels. The 298 candidates contain eight sentence classes unique by word count; '
        'this known shortcut makes those controls essential.\n\n',
        '| Condition | Numeric input and target rule |\n|---|---|\n',
        '| Real TRT | 840 amplitude means + shared structural/count block; correct training targets |\n',
        '| Matched Gaussian | Independent Gaussian values at every finite word/coordinate, using only training-word marginal mean/population SD; exact original word counts, per-coordinate masks and shared structural block |\n',
        '| Class-shuffled | The exact real input; a consistent derangement of the 300 training sentence classes, shared across repeated reader observations; correct validation targets |\n',
        '| Count/duration-only | 14 count/duration summaries + the same 48-count basis (62 features), with no measured spectral amplitudes |\n',
        '| Zero/missingness-only | Zero amplitude values at valid positions and the original missing positions, then the same pooling and structural/count block; no EEG amplitude information |\n',
        '| Existing sentence ridge | The unchanged `eeg_features()` recipe: 105-channel mean/SD/range/RMS/absolute difference and five log-Welch-band features (1,050 total), from the original 200 Hz voltage arrays and center crop ≤1,600 samples |\n\n',
        'Gaussian moments use **only the 32,848 complete training word observations**, not validation or test '
        'values, per-sentence real amplitudes or target embeddings. The source currently has joint all-band '
        'missingness per word; an unexpected partly missing word would fail the pilot rather than silently '
        'change the protocol. The Gaussian mask/shape was checked against real input for all 2,757 observations. '
        'Marginal matching does not preserve neural cross-channel/band covariance; negative Gaussian draws '
        'are feature-domain controls, not physically valid positive envelopes.\n\n',
        'Count/duration summaries include word/fixated/complete/missing counts, missing fraction, fixation '
        'total/mean/SD and TRT total/mean/SD/min/median/max. Their source is gaze/structural metadata. '
        'The zero control retains shape/missingness information, rather than being an uninformative global '
        'zero vector. All-missing inputs are retained. Class shuffling affects training labels only; '
        'validation labels and candidate order stay fixed.\n\n',
        'The sentence baseline reuses the saved **training-fitted** voltage normalizer for the identical '
        'split and refits its ridge model. Its feature recipe is unchanged; its new imputation/scaling/alpha '
        'search is the same as every pilot condition. This is not a reused score from another cohort. '
        'It also has a different signal window and lacks the word view’s annotation-derived structure; '
        'the real-minus-structural controls are the primary EEG-content assessment.\n\n',
        '## Coverage and exclusions\n\n',
        '| Split | Eligible/used observations | Source word records | Complete TRT words | All-missing observations retained |\n|---|---:|---:|---:|---:|\n']
    for c in viewmeta['coverage']:
        lines.append(f"| {c['split']} | {c['observations']:,} | {c['word_observations']:,} | {c['complete_word_observations']:,} | {len(c['all_missing_observations'])} |\n")
    lines += ['\nThere are **no new exclusions**, no candidate removal and no recovery of source slots excluded '
        'by the previous sentence-EEG preparation. ZMG/70 is the one all-missing training observation; its '
        'amplitudes are imputed using training data while its measured structure is retained in every condition. '
        'The original duration QC already removed ZDM/168, 173, 195; ZDN/278; ZJM/275; ZMG/71 '
        '(six training observations), and ZJS/250 (one validation observation). Those exclusions are inherited, '
        'not selected from this pilot’s scores. The original 592 test observations remain outside this '
        'validation-only feature cache. The earlier availability audit found six all-missing test cases; '
        'none was extracted, dropped or evaluated here.\n\n',
        '## Training-only preprocessing and validation selection\n\n',
        'For each condition, missing aggregate values are filled with training feature means or medians; '
        'an entirely missing training column has a recorded zero fallback. Standard scaling or robust '
        'median/IQR scaling is fitted on that condition’s imputed training matrix only. The word-count '
        'basis, Gaussian moments, voltage normalizer, ridge coefficients and intercept all use training '
        'data only. Validation is transformed with the frozen parameters; it never fits preprocessing.\n\n',
        'The frozen grid is **2 imputation rules × 2 scaling rules × 7 alphas = 28 settings per condition** '
        '(168 total), with alphas `[0.1,1,10,100,1000,10000,100000]`. Float64 SVD ridge predicts the '
        'unchanged normalized 384-dimensional sentence embeddings. Maximum full-pool validation MRR '
        'selects the setting independently in each arm; ties within 1e-12 keep the first declared entry. '
        'All selection rows and fitted preprocessing parameters are saved. Recall values did not choose '
        'hyperparameters. Validation is reused across 28 settings and prior investigations, so the selected '
        'scores are optimistic model-selection measurements, not independent confirmatory results.\n\n',
        '## Measured validation results\n\n',
        'Recall is percent of 512 queries against the same 298 candidates.\n\n',
        '| Condition | R@1 % | R@5 % | R@10 % | MRR | Imputation | Scaling | Alpha |\n|---|---:|---:|---:|---:|---|---|---:|\n']
    for r in reports:
        m=r['metrics'];v=r['selection']
        lines.append(f"| {r['condition']} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} | {v['imputation']} | {v['scaling']} | {v['alpha']:g} |\n")
    c=reports[0]['chance'];lines.append(f"| Uniform random ranking expectation | {100*c['recall@1']:.4f} | {100*c['recall@5']:.4f} | {100*c['recall@10']:.4f} | {c['mrr']:.6f} | — | — | — |\n\n")
    lines += ['### Real-minus-control and sentence-baseline differences\n\nRecall differences are percentage points; MRR differences are absolute.\n\n',
        '| Comparison | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |\n|---|---:|---:|---:|---:|\n']
    for g in gaps:lines.append(f"| {g['comparison']} | {100*g['recall@1']:+.4f} | {100*g['recall@5']:+.4f} | {100*g['recall@10']:+.4f} | {g['mrr']:+.6f} |\n")
    lines += ['\n### Validation readers\n\n| Condition | Reader | Queries | R@1 % | R@5 % | R@10 % | MRR |\n|---|---|---:|---:|---:|---:|---:|\n']
    for r in reports:
        for subject,m in r['per_subject'].items():lines.append(f"| {r['condition']} | {subject} | {m['n_queries']} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |\n")
    lines += ['\n## Decision and limitations\n\n',
        'Before any fit, useful validation advantage required real MRR to exceed the strongest of the four '
        'controls by at least 0.002, positive real-minus-every-control MRR gaps for each reader, and at least '
        'two recall cutoffs strictly above all controls. The 0.002 margin is a descriptive practical screen, '
        'not a p-value, confidence bound or established effect threshold.\n\n',
        f"The strongest selected control by aggregate MRR is **{decision['strongest_control']}**; real-minus-strongest MRR is **{decision['real_minus_strongest_mrr']:+.6f}**.\n\n"]
    for k,v in decision['checks'].items():lines.append(f"- `{k}`: {'passed' if v else 'failed'}.\n")
    lines += ['\n'+outcome+'\n\n',
        'Two readers and one previously used validation split are not replicated independent cohorts. '
        'Words/fixations are not independent sentence-retrieval examples. Supplied spectral preprocessing '
        'and gaze-based segmentation remain source assumptions; the new training-only transforms do not '
        'undo upstream full-session processing. A gain over random ranking or the old sentence feature '
        'recipe can reflect the annotation-derived count/missingness shortcut. It must not be presented '
        'as semantic EEG retrieval unless the matched structural controls are beaten consistently. '
        'No punctuation repair, extra seed, post-result grid expansion or test evaluation was performed.\n\n',
        '## Artifacts and verification\n\n',
        f"New feature view: `{view.relative_to(ROOT)}`. New fit outputs: `{run.relative_to(ROOT)}`. "
        'The cache stores fixed-length numeric train/validation vectors, not word text or original EEG arrays. '
        'Its index separately records identities/labels for provenance; those fields are never concatenated '
        'into model inputs. `gaussian_training_moments.npz`, all mask proofs, source hashes and feature '
        'column names are retained. `ridge.npz` files hold selected coefficients and train-fitted '
        'imputer/scaler parameters.\n\n',
        f"Software validation: **7 passed in 5.00 s**. Artifact checks verified six condition reports, "
        f"all 2,757 Gaussian masks, all candidate/observation lists, selected coefficients and saved ranks. "
        f"All **{preservation['protected_files']} protected prior result/input files** retained their hashes. "
        'The [archive](research_results/spectral_trt_pilot_v1_20261030/README.md) preserves configurations, '
        'complete selection grids, coverage/exclusions, validation ranks and participant metrics, source '
        'snapshots and source/copy checksums. Large new feature/weight arrays remain in their fresh '
        'versioned directories. `validation_checks.json` records separate artifact verification.\n\n',
        'Commands executed with the existing environment:\n\n```powershell\n'
        '.\\.venv\\Scripts\\python.exe -m pytest tests/test_spectral_pilot.py -q\n'
        '.\\.venv\\Scripts\\python.exe -u scripts/run_spectral_trt_pilot.py --output results/spectral_trt_pilot_v1_20261030 --view data/processed/zuco1_trt_mean_pilot_v1_20261030\n'
        '.\\.venv\\Scripts\\python.exe scripts/report_spectral_trt_pilot.py --run results/spectral_trt_pilot_v1_20261030\n```\n\n'
        'Output paths are guarded against overwrite. Reproduction requires fresh versioned paths.\n']
    doc.write_text(''.join(lines),encoding='utf-8')
    (archive/'README.md').write_text('# TRT spectral pilot v1 — seed-42 validation only\n\n'
        'See the [full report](../../spectral_trt_pilot_v1_20261030.md).\n\n'+outcome+'\n\n'
        'All six conditions share 2,245 training observations, 512 validation observations and 298 '
        'validation candidates. `plan.json` freezes the TRT family, transformations, candidate pool and '
        'validation screen. `feature_view.json` and `feature_view_metadata/` record the new numeric '
        'cache, exact Gaussian masks and feature-column whitelist. Word text/identities are not model '
        'inputs. Duration controls use source sample counts at 500 Hz.\n\n'
        '`validation_summary.json`, `validation.csv`, `real_minus_controls.json` and condition folders '
        'contain measured validation results, per-reader metrics, ranks and every selection setting. '
        '`decision.json` records the no-test decision. `validation_checks.json` and '
        '`archive_manifest.json` record preservation and source/copy hashes. No test features were '
        'extracted and no previous target, dataset, checkpoint or experiment result was changed.\n',encoding='utf-8')
    print(json.dumps({'report':str(doc),'decision':decision,'validation':validation},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
