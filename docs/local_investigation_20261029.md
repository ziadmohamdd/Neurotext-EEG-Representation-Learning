# Local EEG investigation, 2026-10-29

Executed locally using the existing `.venv` (PyTorch 2.7.1+cu118, NVIDIA
GeForce RTX 3050 Laptop GPU for CNN diagnostics). This is exploratory fixed-pool
sentence retrieval. Cross-release participant independence remains unverified.

## Preservation and scope

Initial `git status --short` contained seven untracked files: `NeuroText_draft.pdf`
and `configs/{alignment_batch8,full_sentence,full_sentence_alignment,
unseen_seed43,unseen_seed44,zjs_existing}.yaml`. They were left in place and not
edited. Starting revision: `003e45cae7cfbd519759007203232cc9a1d23187`.
No checkout, reset, pull, commit, push, dataset rewrite, checkpoint
replacement or previous-result overwrite was performed. Repository/ancestor
instruction searches found no `AGENTS.md`. Guidance reviewed included README,
`continue_work.md`, methodology, the archived research report, transfer
preparation and the published-work comparison. No correspondence was required.

## Training capacity diagnostics

Ran `scripts/run_diagnostics.ps1` with its default 200-step budget. Both datasets
stopped at step 30 under the existing success condition (100% training Recall@1
and loss below 0.05); both first reached 100% at step 20.

| Dataset | Training subject | Initial loss | Final loss | Training Recall@1 | Training MRR |
|---|---|---:|---:|---:|---:|
| ZuCo1 | ZDM | 2.163437 | 0.009686 | 100% | 1.0 |
| ZuCo2 | YAC | 2.162025 | 0.028999 | 100% | 1.0 |

These are eight distinct sentences from one eligible training subject per
version, float32, batch 8, no weight decay or accumulation, with the existing
8-second crop limit. Actual selected durations were 2.735–6.145 s and
2.005–7.165 s. Both audits found no duplicate signals and one flat channel in
each selected observation. Maximum absolute amplitudes were 73.03 and 79.80 µV.
Largest off-diagonal target cosines were 0.7483 and 0.4970, respectively.
The audit does not establish source annotation timing or physiological validity.

Original new outputs: `results/diagnostics/{zuco1,zuco2}_tiny_20261029_232545/`.
After fixing interpreter selection, rerunning without a PATH change reproduced
the same recorded measurements in `*_tiny_20261029_233218/`.
Both sets retain audits, selected manifests, histories and diagnostic checkpoints.
They remain outside research evaluation exports.

The successful tiny mapping argues against an inability of this CNN/loss to
optimize an easy memorization task. It does not establish semantic learning,
generalization, correct original labels, or correctness of every training path.
Noise can also be memorized. The prior batch-2 alignment recipe has only one
negative per example; gradient accumulation does not enlarge that candidate
pool. Existing batch-8/full-sentence experiments also performed poorly, so small
batches alone are not an established explanation.

## Explicitly exploratory transfer

The six seed-42 statistical-ridge jobs use both directions and separate real,
amplitude-matched Gaussian and deranged-label controls. Inputs are the existing
105-channel, 200 Hz, normal-reading duration-QC cohorts, full recordings.
`labram_full_sentence.yaml` here supplies a full-recording configuration; these
are statistical features, not LaBraM features or a new filtering pass.

Used the unchanged manifest and rebuilt shared text cache in
`data/processed/transfer_plan_20261029_213051/`. Fitting signal normalization,
feature scaling and regression uses source training only; source validation
selects regularization. Inclusive target evaluation includes all target subjects
and 50 texts shared with the source release. No target fitting occurs.
All evaluations retain `protocol: transfer_exploratory`,
`identity_audited: false` and the participant-independence limitation. The
original strict-transfer gate and preparation plan were not changed.

Baseline run: `results/exploratory_transfer_20261029_232616/`.
The original grid is `[0.1, 1, 10, 100]`.

| Direction | Training condition | Selected alpha | Recall@1 | MRR |
|---|---|---:|---:|---:|
| ZuCo1→ZuCo2 | Real | 100 | 0.340% | 0.018732 |
| ZuCo1→ZuCo2 | Gaussian | 100 | 0.306% | 0.019507 |
| ZuCo1→ZuCo2 | Shuffled | 1 | 0.340% | 0.019001 |
| ZuCo2→ZuCo1 | Real | 10 | 0.328% | 0.022941 |
| ZuCo2→ZuCo1 | Gaussian | 100 | 0.448% | 0.024007 |
| ZuCo2→ZuCo1 | Shuffled | 10 | 0.328% | 0.021036 |

All six jobs completed and exported to the run's `retrieval.csv`. Baseline real
EEG MRR is below Gaussian MRR in both directions.

Target query/candidate counts are 5,884/344 for ZuCo1→ZuCo2 and 3,349/300
for ZuCo2→ZuCo1. Uniform random-ranking MRR is 0.018661 and 0.020942,
respectively. These chance values do not replace matched negative controls:
Gaussian inputs retain observation duration and per-channel amplitude.

## Justified changes and measured ablation

1. `run_diagnostics.ps1` resolves the repository's `.venv/Scripts/python.exe`
   directly and checks that it exists. The first attempt encountered Windows
   script execution policy; a process-local `-ExecutionPolicy Bypass` was used.
   The sandbox also blocked access to the base Python installation outside the
   workspace; approved execution resolved that without modifying the environment.
2. Ridge selection now shares one train-standardized, centered float64 SVD
   across alphas and accepts `ridge_alphas`. The default four-value grid is
   unchanged. Existing artifact keys and evaluation behavior are retained.
   Synthetic equivalence tests cover rank deficiency and both tall/wide matrices.
3. The first real transfer fit selected the old upper boundary, α=100. Tested
   `[0.1, 1, 10, 100, 1000, 10000, 100000]` in a separately named ablation,
   using source-validation MRR only. `scripts/refit_ridge_path.py` verifies
   original source split digests and the text-cache digest, reuses freshly
   computed local feature caches, records cache hashes and lineage, and writes
   fresh artifacts. No target metric chooses alpha or the grid winner.

On the 2,757×1,050 real ZuCo1 training feature matrix, four independent sklearn
fits took 3.1231 s versus 0.8026 s for the shared-SVD path: 3.89× faster fitting.
Maximum coefficient difference was 1.01×10⁻¹². This is one local timing and
excludes feature extraction/evaluation; it is not a 3.89× end-to-end speed claim.
Benchmark code and measurements are in
`results/exploratory_regularization_20261029/{benchmark.py,benchmark.json}`.

| Direction | Training condition | Selected alpha | Recall@1 | MRR |
|---|---|---:|---:|---:|
| ZuCo1→ZuCo2 | Real | 100 | 0.340% | 0.018732 |
| ZuCo1→ZuCo2 | Gaussian | 100 | 0.306% | 0.019507 |
| ZuCo1→ZuCo2 | Shuffled | 100,000 | 0.272% | 0.018177 |
| ZuCo2→ZuCo1 | Real | 1,000 | 0.627% | 0.024933 |
| ZuCo2→ZuCo1 | Gaussian | 1,000 | 0.448% | 0.023354 |
| ZuCo2→ZuCo1 | Shuffled | 100,000 | 0.299% | 0.020941 |

All six ablations completed. Artifacts are in
`results/exploratory_regularization_20261029/`; the real forward subfolder is
`zuco1_to_zuco2_real`, and the remaining five retain the baseline job names.
`summary.csv` and `summary.json` contain all 12 evaluations, validation scores,
selected alphas, candidate counts and report paths. The saved `summarize.py`
asserts all baseline jobs completed, all evaluations retain exploratory identity
disclosures and 50 shared texts, and both grids use matching candidate hashes
and query counts within each condition.

The real ZuCo2→ZuCo1 alpha changes from 10 to 1,000. Source-validation MRR
increases from 0.023165 to 0.026690; target Recall@1 rises from 11/3,349
(0.328%) to 21/3,349 (0.627%), and target MRR from 0.022941 to 0.024933
(8.68% relative). The real forward result is unchanged because validation
still selects alpha 100. This is a measured improvement on one exploratory split,
not evidence of reliable semantic transfer.

For the wider grid, reverse real EEG has 21 top-1 hits versus 15 Gaussian and
10 shuffled hits. Its MRR advantage over Gaussian is 0.001579. Forward real EEG
still has lower MRR than Gaussian. Wider-grid validation improvements also
reduce target MRR for Gaussian reverse and both shuffled controls, illustrating
that source-validation gains need not transfer. Both shuffled fits select the
new upper boundary; this is a control outcome, not a reason to search target
scores for a more favorable model. The wider grid remains an explicit ablation,
not a silently substituted historical baseline or a validated new default.

Reproduce fresh ablations from the completed baseline caches with:

```powershell
.\.venv\Scripts\python.exe scripts/refit_ridge_path.py --original results/exploratory_transfer_20261029_232616/0003_ridge_exploratory_zuco2_to_zuco1_real_q0_s42 --manifest data/processed/transfer_plan_20261029_213051/combined.jsonl --text-cache data/processed/transfer_plan_20261029_213051/sentence_embeddings.npz --config configs/transfer_ridge_regularized.yaml --output results/NEW_UNIQUE_ABLATION_FOLDER
```

For fresh feature computation rather than cached refitting, use the existing
`run-experiments` command in `continue_work.md` with
`--config configs/transfer_ridge_regularized.yaml` and a new output folder.

## Validation and remaining limits

Initial investigation suite (before the three-seed extension): **130 passed,
2 skipped, 1 deselected**, with two pre-existing
short-signal matplotlib warnings. Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -k 'not test_file_above_old_limit_without_full_materialization'
```

An earlier full suite passed 128 tests (before the cached-ablation tests were
added). A later full run passed 128 and failed the large-MAT fixture with
`OSError: [Errno 28] No space left on device`; only about 328 MiB remained,
while that fixture writes a 513 MiB file on this filesystem. That earlier exclusion
is explicit; the large-file test was not modified or called successful in the
initial suite. The two skipped tests are optional-environment tests. Cached-refit
tests cover real/Gaussian/shuffled controls, equality with fresh fits, refusal
to overwrite outputs, and byte-for-byte preservation of original artifacts.

The initial transfer comparison used seed 42; the three-seed extension below
adds seeds 43 and 44 for statistical ridge. No new multi-seed neural
study, source-unseen-text transfer evaluation, or independently verified
participant-disjoint transfer is established. The candidate pool includes shared
texts. Existing timing, reference/filter provenance and approximate pretrained
montage limitations remain. Small hit counts and repeated stimuli limit inference;
no statistical significance or competitive EEG-to-text generation claim is made.
The archived research-results export remains unchanged; these new results have
separate paths and explicit protocols.

## Wider-grid repetition across seeds 42, 43 and 44

The follow-up fixes the seven-value grid in advance and repeats both directions
with real, Gaussian and shuffled inputs for seeds 43 and 44. These are 12 new
fits, alongside the six existing wider-grid seed-42 fits. The original four-value
seed-42 baseline is excluded from this three-seed comparison. No alpha-grid
extension, target-driven model selection, data preparation or new identity claim
was introduced for the repeats. Statistical features are recomputed for each
new source split/control seed using the unchanged manifest and shared text cache.

Fresh run folders:

- `results/exploratory_wider_seed43_20261029/`
- `results/exploratory_wider_seed44_20261029/`
- Existing wider-grid seed 42: `results/exploratory_regularization_20261029/`

Each new job retains `metadata.json` with the full effective config, source
normalizer, validation scores for all seven alphas, selected alpha, source/target
release labels, candidate keys, text-cache digest and full split manifests.
`evaluation.json` retains target candidate keys/hash, query count, per-subject
metrics and exploratory identity disclosures. `ridge.npz`, training/validation
feature caches and `splits.json` remain beside it. Each run root has `jobs.json`
and its own `retrieval.csv`. The two worker processes use the existing `.venv`;
the training implementation is unchanged from the preceding investigation.

Commands (both require new output folders when repeated):

```powershell
.\.venv\Scripts\python.exe -m eeg_decoding.cli run-experiments --manifest data/processed/transfer_plan_20261029_213051/combined.jsonl --text-cache data/processed/transfer_plan_20261029_213051/sentence_embeddings.npz --config configs/transfer_ridge_regularized.yaml --experiments configs/transfer_exploratory_seed43.yaml --output results/exploratory_wider_seed43_20261029 --execute
.\.venv\Scripts\python.exe -m eeg_decoding.cli run-experiments --manifest data/processed/transfer_plan_20261029_213051/combined.jsonl --text-cache data/processed/transfer_plan_20261029_213051/sentence_embeddings.npz --config configs/transfer_ridge_regularized.yaml --experiments configs/transfer_exploratory_seed44.yaml --output results/exploratory_wider_seed44_20261029 --execute
```

### Summary audit and interpretation

`eeg_decoding.evaluation.transfer_summary` requires exactly 18 evaluations:
two directions × three seeds × three controls. It checks the grid and first
source-validation-MRR maximum, source training/validation release membership,
disjoint recorded split labels, training-only normalizer membership, and unchanged
effective recipes apart from seed. It also checks matching source split digests
within each seed, target query digests and candidate pools across seeds/controls,
matching text caches, appropriate evaluation controls, unverified identities and
the actual source/target sentence overlap. These metadata checks accompany the
existing training code's source-only fitting/selection; they cannot independently
verify cross-release person identity or original EEG annotations.

The summary records all run paths, selected alphas and validation scores,
configurations, source training/validation subjects, target subjects, validation
candidate keys, target keys/hash, and input report/metadata checksums. It refuses
to overwrite a prior summary. Means give each seed equal weight; SD is sample SD
with denominator 2 across the three runs. Paired differences subtract Gaussian
from real within the same direction and seed before calculating mean and SD.
Recall differences are expressed in percentage points.

All three seeds evaluate the **same target observations, participants and
sentence pools** in each direction. Source participants are reused in overlapping
training/validation splits; noise and shuffled-label draws also change with seed.
The runs are repeated split/control realizations, not three independent cohorts.
Their SD is descriptive variation, not a confidence interval or standard error.
No pooled query sample size, independence-based significance test, or claim of
verified unseen-person/unseen-text transfer follows from these repetitions.

### Measured three-seed findings

Both new run roots finished all six jobs with status `executed` and exported
their own six-row retrieval CSVs. The summary audit passed for all 18 wider-grid
evaluations. SHA256 verification of 67 files in the previous baseline and
regularization folders found **zero changes**; the before-check hashes are in
`results/exploratory_repeat_prior_hashes_20261029.json`. Datasets and prior
checkpoints were not rewritten.

Per-seed target MRR and selected alpha (R/G/S means real/Gaussian/shuffled):

| Direction | Seed | Selected alpha R/G/S | Real MRR | Gaussian MRR | Shuffled MRR |
|---|---:|---|---:|---:|---:|
| ZuCo1→ZuCo2 | 42 | 100 / 100 / 100,000 | 0.018732 | 0.019507 | 0.018177 |
| ZuCo1→ZuCo2 | 43 | 1,000 / 0.1 / 0.1 | 0.019005 | 0.020673 | 0.017791 |
| ZuCo1→ZuCo2 | 44 | 0.1 / 10 / 100 | 0.017851 | 0.017859 | 0.018772 |
| ZuCo2→ZuCo1 | 42 | 1,000 / 1,000 / 100,000 | 0.024933 | 0.023354 | 0.020941 |
| ZuCo2→ZuCo1 | 43 | 1,000 / 1 / 1,000 | 0.024221 | 0.022323 | 0.021521 |
| ZuCo2→ZuCo1 | 44 | 100,000 / 10 / 1 | 0.021621 | 0.023748 | 0.021185 |

Equal-weight means ± sample SD across the three seeds:

| Direction | Condition | Recall@1 (%) ± split SD (pp) | MRR ± split SD |
|---|---|---:|---:|
| ZuCo1→ZuCo2 | Real | 0.278 ± 0.055 | 0.018529 ± 0.000603 |
| ZuCo1→ZuCo2 | Gaussian | 0.346 ± 0.132 | 0.019346 ± 0.001414 |
| ZuCo1→ZuCo2 | Shuffled | 0.249 ± 0.020 | 0.018247 ± 0.000494 |
| ZuCo2→ZuCo1 | Real | 0.508 ± 0.119 | 0.023592 ± 0.001743 |
| ZuCo2→ZuCo1 | Gaussian | 0.458 ± 0.105 | 0.023142 ± 0.000736 |
| ZuCo2→ZuCo1 | Shuffled | 0.368 ± 0.062 | 0.021216 ± 0.000291 |

Paired real-minus-Gaussian differences, using unrounded metrics:

| Direction | Seed | ΔRecall@1 (pp) | ΔMRR |
|---|---:|---:|---:|
| ZuCo1→ZuCo2 | 42 | +0.033990 | −0.000775 |
| ZuCo1→ZuCo2 | 43 | −0.237933 | −0.001668 |
| ZuCo1→ZuCo2 | 44 | 0.000000 | −0.000008 |
| ZuCo2→ZuCo1 | 42 | +0.179158 | +0.001579 |
| ZuCo2→ZuCo1 | 43 | +0.149298 | +0.001898 |
| ZuCo2→ZuCo1 | 44 | −0.179158 | −0.002127 |

| Direction | Mean ΔRecall@1 ± split SD (pp) | Mean ΔMRR ± split SD | Positive ΔMRR seeds |
|---|---:|---:|---:|
| ZuCo1→ZuCo2 | −0.067981 ± 0.148161 | −0.000817 ± 0.000831 | 0/3 |
| ZuCo2→ZuCo1 | +0.049766 ± 0.198815 | +0.000450 ± 0.002237 | 2/3 |

The forward real MRR is below Gaussian on every seed. Reverse real MRR is higher
on seeds 42 and 43, but the sign reverses on seed 44. Its small positive average
difference is accompanied by substantially larger descriptive split/control
variation. The original seed-42 regularization gain remains a measured result,
but the three-seed comparison does **not establish a consistent real-EEG
advantage**. Source-validation alpha selection is variable: reverse real EEG
selects the fixed upper boundary 100,000 on seed 44, while its target MRR is the
lowest of the three real reverse runs. No further grid search was performed.

Target pools and observations are identical across all nine runs per direction:

| Direction | Target queries | Target subjects | Unique candidates | Shared source texts |
|---|---:|---:|---:|---:|
| ZuCo1→ZuCo2 | 5,884 | 18 | 344 | 50 |
| ZuCo2→ZuCo1 | 3,349 | 12 | 300 | 50 |

Pool SHA256 values (hashing the saved ordered key lists):

- ZuCo1→ZuCo2: `57bbe847741650bb398d2c6964e8f69282cf968910978ca0173f8cfca6f4cca2`.
- ZuCo2→ZuCo1: `33787a9cf7129e1a5c37160ab7bf963ff257743f1c5d81725da62172719f1334`.

The complete summary is in `results/exploratory_three_seed_summary_20261029/`:

- `runs.csv`: all 18 unrounded measurements, source-validation winners/scores,
  split digests, candidate hashes, cache digest, identity flags and report paths.
- `condition_summary.csv`: means and descriptive sample SD by condition.
- `paired_differences.csv` and `difference_summary.csv`: per-seed paired
  differences and their descriptive means/SD.
- `summary.json`: the same results with the repeated-cohort interpretation.
- `configurations_and_pools.json`: full effective configurations, source/target
  subject lists, validation selection tables, source-validation candidate keys,
  target candidate keys and report/metadata checksums for each run.

Rebuild into a fresh summary folder with:

```powershell
.\.venv\Scripts\python.exe -m eeg_decoding.evaluation.transfer_summary --roots results/exploratory_regularization_20261029 results/exploratory_wider_seed43_20261029 results/exploratory_wider_seed44_20261029 --output results/NEW_UNIQUE_THREE_SEED_SUMMARY
```

### Exact follow-up test exclusions

Full-suite command: `.\.venv\Scripts\python.exe -m pytest -q -rs`.
Result: **139 passed, 2 skipped, 0 deselected**, with the same two existing
short-signal matplotlib warnings. This includes eight new summary tests for
paired statistics/sample SD, incomplete/duplicate runs, inconsistent pools/splits,
wrong alpha selection, identity disclosure, normalizer leakage, target records
in source training, grid mismatch and preservation of input artifacts.

The only skips are:

- `tests/test_storage_and_pretrained.py::test_official_pretrained_weights_on_synthetic_input`:
  `EEG_LABRAM_CACHE` is unset; pytest reason is
  `optional official-weight integration; no network download during unit tests`.
- `tests/test_storage_and_pretrained.py::test_pretrained_cache_extraction`:
  `EEG_LABRAM_CACHE` is unset; pytest reason is
  `optional official-weight extraction on synthetic fixtures`.

The earlier deselection was exactly
`tests/test_mat5_streaming.py::test_file_above_old_limit_without_full_materialization`.
About 5.8 GiB was free at the start of this follow-up, so that 513 MiB test
was included and **passed**. There is no disk-space test exclusion in this run.
The prior disk-limited run remains documented above as historical evidence.
