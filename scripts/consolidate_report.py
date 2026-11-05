"""Update the report from archived tables only; no experiment execution."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCH=ROOT/'docs/research_results/wider_grid_transfer'


def section():
    data=json.loads((ARCH/'summary.json').read_text(encoding='utf-8'))
    labels={'zuco1_to_zuco2':'ZuCo1 → ZuCo2','zuco2_to_zuco1':'ZuCo2 → ZuCo1'}
    lines=['## Wider-grid exploratory cross-dataset transfer', '',
        'The 18 wider-grid evaluations are **separate from the original-grid experiments**. '
        'They cover both directions, seeds 42/43/44 and real, Gaussian and shuffled '
        'statistical-ridge conditions. The original grid `[0.1, 1, 10, 100]` was '
        'run only at seed 42 for transfer. The wider grid is fixed at '
        '`[0.1, 1, 10, 100, 1000, 10000, 100000]`. Do not pool these grids or '
        'treat the original-grid runs as extra repeats.', '',
        'Inputs are the duration-QC 105-channel, 200 Hz normal-reading cohorts '
        'with full recordings. These are statistical features, not the 19-channel '
        'LaBraM adaptation. Normalization, feature scaling and ridge fitting use '
        '**source training observations only**. Source-validation MRR selects '
        'alpha, with the first maximum winning ties; target scores never select '
        'alpha. Gaussian validation/evaluation uses Gaussian inputs, while shuffled '
        'training uses true validation/test labels.', '',
        'Every run retains `protocol: transfer_exploratory` and '
        '`identity_audited: false`. Cross-release participant independence is '
        '**unresolved**. Different subject prefixes do not verify different people. '
        'Inclusive target pools contain 50 texts shared with the source release; '
        'these results establish neither unseen-person nor unseen-text transfer. '
        'The strict transfer identity gate remains unchanged and strict transfer '
        'is unexecuted.', '',
        '| Direction | Queries per run | Target subjects | Unique candidates | Uniform-ranking MRR |',
        '|---|---:|---:|---:|---:|',
        '| ZuCo1 → ZuCo2 | 5,884 | 18 | 344 | 0.018661 |',
        '| ZuCo2 → ZuCo1 | 3,349 | 12 | 300 | 0.020942 |', '',
        'Target query digests and ordered candidate hashes match across all nine '
        'runs per direction. The same target observations and participants recur '
        'at every seed. Source splits overlap and noise/shuffle draws change with '
        'seed. Equal-weight means and sample SD across three seeds describe this '
        'variation; **repeated splits are not independent cohorts**. SD is not a '
        'confidence interval or standard error. Gaussian inputs retain duration '
        'and per-channel amplitude, so uniform-ranking chance cannot substitute '
        'for these matched controls.', '',
        '### Seed-level wider-grid results', '',
        '| Direction | Seed | Condition | Selected alpha | Source validation MRR | Target Recall@1 | Target MRR |',
        '|---|---:|---|---:|---:|---:|---:|']
    for d in labels:
        for seed in [42,43,44]:
            for c in ['real','gaussian','shuffled']:
                r=next(r for r in data['runs'] if (r['direction'],r['seed'],r['control'])==(d,seed,c))
                lines.append(f"| {labels[d]} | {seed} | {c.capitalize()} | {r['selected_alpha']:g} | {r['validation_mrr']:.6f} | {100*r['recall_at_1']:.3f}% | {r['mrr']:.6f} |")
    lines += ['', '![Seed-level wider-grid MRR](wider_grid_transfer/wider_grid_seed_mrr.png)', '',
        'Figure: all 18 wider-grid runs. Dashed lines show uniform-ranking expectations, '
        'not significance thresholds. Full-precision metrics are in '
        '[runs.csv](wider_grid_transfer/runs.csv).', '',
        '### Descriptive means and paired controls', '',
        '| Direction | Condition | Mean Recall@1 ± split SD (percentage points) | Mean MRR ± split SD |',
        '|---|---|---:|---:|']
    for r in data['condition_summary']:
        lines.append(f"| {labels[r['direction']]} | {r['control'].capitalize()} | {100*r['mean_recall_at_1']:.3f}% ± {100*r['split_sd_recall_at_1']:.3f} | {r['mean_mrr']:.6f} ± {r['split_sd_mrr']:.6f} |")
    lines += ['', '![Wider-grid means and SD](wider_grid_transfer/wider_grid_mean_sd.png)', '',
        'Figure: bars are equal-weight means, error bars are sample SD (ddof=1), '
        'and points are individual seeds. These reuse target cohorts.', '',
        '| Direction | Seed | Real − Gaussian Recall@1 (pp) | Real − Gaussian MRR |',
        '|---|---:|---:|---:|']
    for r in data['paired_differences']:
        lines.append(f"| {labels[r['direction']]} | {r['seed']} | {r['real_minus_gaussian_recall_at_1_pp']:+.6f} | {r['real_minus_gaussian_mrr']:+.6f} |")
    lines += ['', '| Direction | Mean ΔRecall@1 ± SD (pp) | Mean ΔMRR ± SD | Positive ΔMRR seeds |',
        '|---|---:|---:|---:|']
    for r in data['difference_summary']:
        lines.append(f"| {labels[r['direction']]} | {r['mean_recall_at_1_difference_pp']:+.6f} ± {r['split_sd_recall_at_1_difference_pp']:.6f} | {r['mean_mrr_difference']:+.6f} ± {r['split_sd_mrr_difference']:.6f} | {r['positive_mrr_seeds']}/3 |")
    lines += ['', '![Paired wider-grid differences](wider_grid_transfer/wider_grid_real_minus_gaussian.png)', '',
        'Differences pair real and Gaussian within direction and seed before '
        'calculating their mean and sample SD. Forward real MRR is below Gaussian '
        'on all three seeds. Reverse real MRR is higher on seeds 42 and 43 and '
        'lower on seed 44; its small positive mean difference is accompanied by '
        'substantially larger descriptive variation. **These configurations do '
        'not demonstrate reliable retrieval beyond controls.** No significance '
        'claim, pooled independent query count, or general claim about the absence '
        'of semantic information in EEG follows.', '',
        '### Original grid versus wider grid: seed 42 only', '',
        '| Direction | Condition | Original alpha / MRR | Wider alpha / MRR |',
        '|---|---|---:|---:|']
    original=list(csv.DictReader((ARCH/'original_grid_seed42.csv').open()))
    for d in labels:
        for c in ['real','gaussian','shuffled']:
            old=next(r for r in original if r['direction']==d and r['control']==c)
            new=next(r for r in data['runs'] if (r['direction'],r['seed'],r['control'])==(d,42,c))
            lines.append(f"| {labels[d]} | {c.capitalize()} | {float(old['selected_alpha']):g} / {float(old['mrr']):.6f} | {new['selected_alpha']:g} / {new['mrr']:.6f} |")
    lines += ['', 'The seed-42 reverse real-EEG regularization gain remains a measured '
        'result: Recall@1 rises from 11/3,349 (0.328%) to 21/3,349 (0.627%), '
        'and MRR from 0.022941 to 0.024933. It is not a three-seed original-versus-wider '
        'ablation: the original grid was not repeated for seeds 43/44. '
        'Wider-grid selection varies by seed/control and reverse real EEG chooses '
        'the fixed upper boundary at seed 44. No additional grid search was run.', '',
        '### Archive, provenance and validation', '',
        'The original 79-row [retrieval.csv](retrieval.csv) is unchanged and '
        'contains neither these 18 wider-grid transfer evaluations nor the six '
        'original-grid transfer comparisons. They remain in separate archives. '
        'The [archive manifest](wider_grid_transfer/archive_manifest.json) lists '
        'source paths and SHA256 checksums. Archived artifacts include the '
        '[18-run summary](wider_grid_transfer/summary.json), '
        '[configurations and candidate pools](wider_grid_transfer/configurations_and_pools.json), '
        'per-condition statistics, paired differences, the six original-grid '
        'comparisons, copied evaluation JSONs, exact YAML recipes and implementation '
        'snapshots. Original datasets, checkpoints and experiment files are unchanged. '
        'Consolidation only reads stored metrics and renders artifacts; it launches '
        'no additional experiments.', '',
        'Recorded validation: **139 passed, 2 optional skips, 0 deselected**, '
        'from `.venv/Scripts/python.exe -m pytest -q -rs`. This is the recorded '
        'pre-consolidation software suite, not a newly executed experiment or '
        'a claim that consolidation reran the suite. Both optional tests skip '
        'because `EEG_LABRAM_CACHE` is unset:', '',
        '- `tests/test_storage_and_pretrained.py::test_official_pretrained_weights_on_synthetic_input`: '
        '`optional official-weight integration; no network download during unit tests`.',
        '- `tests/test_storage_and_pretrained.py::test_pretrained_cache_extraction`: '
        '`optional official-weight extraction on synthetic fixtures`.', '',
        'The formerly disk-limited '
        '`tests/test_mat5_streaming.py::test_file_above_old_limit_without_full_materialization` '
        'was included and passed. The two existing short-signal matplotlib warnings '
        'remain documented. No disk-space exclusion applies to the recorded 139-test '
        'validation. Rendering/package/data-consistency checks for this consolidation '
        'are separate artifact checks.', '']
    return '\n'.join(lines)


if __name__=='__main__':
    target=ROOT/'docs/research_results/REPORT.md'
    content=target.read_text(encoding='utf-8')
    if '## Wider-grid exploratory cross-dataset transfer' in content:
        raise FileExistsError('Transfer section already inserted; preserve the existing consolidation')
    target.write_text(content.replace('## Completion checklist',section()+'\n## Completion checklist'),encoding='utf-8')
    print('Inserted audited wider-grid section into the main report')
