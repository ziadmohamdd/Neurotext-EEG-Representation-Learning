# Wider-grid exploratory transfer archive

This archive contains **18 wider-grid evaluations**: seeds 42, 43 and 44,
two directions, and real/Gaussian/shuffled statistical-ridge conditions.
The six original-grid seed-42 transfer comparisons are separate in
`original_grid_seed42.csv` and `original_grid_evaluations/`. Neither set is
merged into the original 79-row `../retrieval.csv`.

The wider alpha grid is `[0.1, 1, 10, 100, 1000, 10000, 100000]`; the original
grid stops at 100. Source training fits normalization/scaling/regression, and
source-validation MRR selects alpha. Target observations never select alpha.
All runs are explicitly exploratory, with cross-release participant independence
unresolved and 50 shared source/target sentence texts retained in the pools.

`summary.json`, the four summary CSVs and `configurations_and_pools.json` were
copied unchanged from the completed three-seed analysis. The latter records full
effective configurations, source/target subjects, validation selection tables,
candidate lists, normalizers and source report/metadata checksums. `runs.csv`
records full-precision measurements, selected alphas, split/cache/pool digests
and source report paths. Copied evaluation JSONs are in `evaluations/`.
Exact YAML recipes and code snapshots are in `configs/` and `implementation/`.

Figures are exported as PNG and SVG:

- `wider_grid_seed_mrr`: all seeds and controls, with uniform-ranking references.
- `wider_grid_mean_sd`: equal-weight means, sample SD and individual seed points.
- `wider_grid_real_minus_gaussian`: paired differences and their mean/sample SD.

SD is descriptive across repeated source splits/control draws. The same target
participants and queries recur across seeds; these are not independent cohorts.
No CI, pooled independent sample size or significance claim is inferred.

Recorded software validation: **139 passed, 2 optional skips, 0 deselected**.
`EEG_LABRAM_CACHE` was unset. Exact skip node IDs/reasons are in the manifest
and main report. The earlier disk-limited MATLAB test was included and passed.
`consolidation_validation.json` records separate archive/package/table checks.
Consolidation launched no experiments.

These configurations do not demonstrate reliable retrieval beyond controls.
See [the main report](../REPORT.md) for tables and interpretation. The previous
report and 14-slide deck are retained in `../../presentation/archive/pre_wider_grid/`.
