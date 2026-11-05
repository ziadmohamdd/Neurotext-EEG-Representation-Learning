# EEG representation learning: results and completion review

Research consolidation, 2026-10-30. Measurements come from local Windows execution
(GPU neural fits and CPU ridge fits). The original 79 reports remain in [retrieval.csv](retrieval.csv);
the 18 wider-grid exploratory transfer evaluations and six separate original-grid
comparisons are archived in [wider_grid_transfer](wider_grid_transfer/archive_manifest.json). This is a
descriptive analysis of executed experiments, not a claim of independent cloud
reproduction or a completed research campaign.

## Main finding

The tested statistical, supervised CNN, masked-reconstruction SSL and frozen
LaBraM representations do not establish reliable sentence retrieval across
subjects or unseen sentence content. Real EEG does not consistently outperform
matched Gaussian controls. Some results exceed uniform random-ranking
expectations, but that alone is insufficient evidence of useful EEG decoding.

Across the 18 wider-grid transfer evaluations, forward real MRR is lower than
Gaussian on all three seeds; reverse differences change sign. These
configurations do not demonstrate reliable retrieval beyond controls.
Participant independence across releases remains unresolved.

The validation-only TRT studies make a separate distinction: structural counts and
durations retrieve sentence classes at MRR 0.292012, while real TRT adds only
+0.000299 over structure and +0.000154 over the Gaussian-combined model. Reader
effects change sign and Recall@1 declines. These selected, repeatedly reused
validation scores do not demonstrate a consistent incremental EEG advantage.
Further tuning and test evaluation are stopped.

These negative findings are scientifically reportable. They do not establish
that EEG contains no semantic information or that pretrained EEG models cannot
work. Results concern these implementations, adaptations, cohorts and splits.

## Data and methods

ZuCo1 preparation produced 3,356 observations from 12 subjects; ZuCo2 produced
5,911 from 18. Original missing EEG slots were audited. Duration QC excluded
7 ZuCo1 and 27 ZuCo2 segments of at most 0.1 seconds, leaving 3,349 and 5,884.
An inspected YTL source slot contained a single sample and was excluded during
technical preparation with an audit entry. The selected condition is normal
reading (`nr`), with 105 ordered channels resampled to 200 Hz. Existing source
preprocessing was retained; anti-alias resampling was applied. Complete original
trigger timing and per-run reference/filter histories have not been verified.

Models predict cached MiniLM sentence embeddings. Evaluation ranks a fixed,
explicit pool of unique sentences with cosine similarity. This is retrieval,
not unrestricted text generation. Data-dependent normalization and feature
scaling use training observations. Model selection uses validation subjects;
test inspection across development makes the overall history exploratory.

Statistical waveform features combine per-channel statistics and spectral band powers.
Ridge uses float64 scaling and SVD fitting. The CNN learns compact waveform
representations. SSL first reconstructs masked temporal blocks, then initializes
sentence alignment. Historical waveform Gaussian controls match each observation's channel standard
deviations and therefore preserve amplitude information; they are not equivalent
to uniformly random rankings. Shuffled controls use separately trained models
with deranged sentence labels and true evaluation labels.

The validation-only spectral Gaussian instead uses training-word marginal
moments and preserves word counts/missingness, as specified in its section.

Earlier ZuCo1 inputs were center-cropped to at most 8 seconds. Later filtered
experiments use complete recordings. These are separate conditions, not a
controlled cropping-only ablation. Repeated neural alignment runs had declining
training loss and worsening validation loss; early stopping selected epoch 0.

## Supervised CNN negative controls

The updated export adds four CNN control evaluations to the original 63 reports.
The original rows and measurements are unchanged. Comparisons below match the
existing real-EEG supervised CNN in each dataset, not the SSL-initialized model.

| Dataset | Training condition | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| ZuCo1 | Real EEG | 0.338% | 1.858% | 3.209% | 0.020596 |
| ZuCo1 | Gaussian inputs | 0.338% | 1.858% | 3.885% | 0.022081 |
| ZuCo1 | Shuffled pairs | 0.169% | 1.689% | 3.716% | 0.020392 |
| ZuCo2 | Real EEG | 0.305% | 0.813% | 2.541% | 0.016582 |
| ZuCo2 | Gaussian inputs | 0.305% | 1.524% | 3.252% | 0.019452 |
| ZuCo2 | Shuffled pairs | 0.203% | 0.915% | 2.134% | 0.016242 |

ZuCo1 uses the original 3,356-observation cohort, an 8-second center crop,
microbatch size 2, 592 queries and 300 candidates. ZuCo2 uses the filtered
5,884-observation cohort, full recordings, microbatch size 8, 984 queries and
344 candidates. Both use seed 42. These two data recipes are not a controlled
cross-dataset performance comparison. The summarizer checks equal candidate
pool hashes and matching protocol, seed and query counts
within each dataset; full preprocessing/configuration matching also relies on
the recorded execution commands because these fields are absent from the CSV.

Real and Gaussian CNNs have equal top-1 hit counts (2 in ZuCo1, 3 in ZuCo2),
and Gaussian MRR is higher in both. The results do not demonstrate a real-EEG
advantage for these supervised CNN configurations. No statistical significance
is claimed from one split. SSL and adapted pretrained ridge controls are
reported separately below; these CNN results must not be relabeled as those
controls.

![Supervised CNN controls](cnn_controls_mrr.png)

The source filenames and full metrics are in
[cnn_controls_summary.csv](cnn_controls_summary.csv).

## Full-sentence SSL negative controls

These comparisons use the duration-filtered 105-channel cohorts (3,349 ZuCo1
and 5,884 ZuCo2 observations), full recordings, seed 42 and alignment microbatch
size 8. They match each full-sentence real-EEG SSL/alignment configuration,
not the earlier cropped ZuCo1 experiment.

| Dataset | Condition | Recall@1 | MRR |
|---|---|---:|---:|
| ZuCo1 | Real EEG | 0.169% | 0.020680 |
| ZuCo1 | Gaussian inputs | 0.338% | 0.023159 |
| ZuCo1 | Shuffled pairs | 0.338% | 0.019842 |
| ZuCo2 | Real EEG | 0.305% | 0.018846 |
| ZuCo2 | Gaussian inputs | 0.305% | 0.018117 |
| ZuCo2 | Shuffled pairs | 0.305% | 0.019402 |

ZuCo1 has 592 queries/300 candidates; ZuCo2 has 984/344. Gaussian MRR exceeds
real EEG on ZuCo1; shuffled MRR exceeds real EEG on ZuCo2. Real top-1 hits are
1 versus 2 for each control in ZuCo1 and 3 for all three conditions in ZuCo2.
These outcomes do not establish a consistent real-EEG advantage. Differences
are descriptive; no statistical significance claim is made.

Gaussian SSL was pretrained from scratch on Gaussian waveforms from training
observations, then used to initialize Gaussian-input sentence alignment. Thus
the Gaussian condition covers both learned stages, rather than corrupting only
the input after real-data pretraining. Shuffled-pair alignment reused the real
SSL encoder because that pretraining objective uses no sentence labels.
Normalization uses only training observations; validation selects checkpoints.

Recorded Gaussian SSL folders end in `20261029_060922`. Best pretraining epochs
were 15 (ZuCo1, validation reconstruction loss 0.444607) and 4 (ZuCo2, 2.002514).
Shuffled alignment initialized from real SSL checkpoints in
`zuco1_ssl_full_sentence_20261027_170627` and
`zuco2_ssl_full_20261028_211015`. Alignment control evaluations end in
`20261029_142036`. Pretraining details come from the user's execution transcript;
the CSV itself does not contain upstream checkpoint identities. Original
checkpoints and configurations should be retained for the final audit.

![SSL controls](ssl_controls_mrr.png)

Full metrics and source filenames are in
[ssl_controls_summary.csv](ssl_controls_summary.csv).

## Matched pretrained comparison

Stored coordinates from 30 recording headers match a HydroCel template.
The documented approximate mapping selects 19 real electrodes for standard
positional tokens, with angular errors of 2.11–9.40 degrees and a Cz anchor.
Both comparison methods use identical adapted signals, complete seconds and
subject splits. Inputs under one second are excluded, leaving 3,345 ZuCo1 and
5,838 ZuCo2 observations. Official frozen LaBraM weights are checksum-verified.
Mapping, upstream filtering and reference uncertainties limit paper-reproduction
claims; this is an exploratory adaptation.

| Dataset | Features + ridge | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| ZuCo1 | Statistical | 0.507% | 1.689% | 3.716% | 0.024530 |
| ZuCo1 | LaBraM | 0.169% | 2.196% | 4.392% | 0.022326 |
| ZuCo2 | Statistical | 0.814% | 2.442% | 4.374% | 0.027599 |
| ZuCo2 | LaBraM | 0.407% | 2.238% | 3.967% | 0.023559 |

ZuCo1 has 592 queries/300 candidates; ZuCo2 has 983/344. Uniform chance MRR is
0.020942 and 0.018661, respectively. LaBraM has lower top-1 and MRR in both
datasets; ZuCo1 Recall@5/10 is higher. This is one split, without significance
claims. Matched adapted-cohort negative controls are reported next.

![Matched representations](pretrained_comparison.png)

## Frozen LaBraM ridge negative controls

These use the same 19-channel prepared observations, whole-second temporal
support, seed 42, subject split and candidate pool as each real-EEG LaBraM ridge.
The completed real feature caches were verified and reused; Gaussian waveforms
were passed through the same frozen official backbone. Separate Gaussian and
shuffled-pair ridge models were then fit with source-training normalization and
validation selection. These are not controls for a fine-tuned LaBraM model or
for the ZuCo SSL encoder.

| Dataset | Condition | Recall@1 | MRR |
|---|---|---:|---:|
| ZuCo1 | Real EEG | 0.169% | 0.022326 |
| ZuCo1 | Gaussian inputs | 0.676% | 0.028627 |
| ZuCo1 | Shuffled pairs | 0.338% | 0.021823 |
| ZuCo2 | Real EEG | 0.407% | 0.023559 |
| ZuCo2 | Gaussian inputs | 0.509% | 0.022214 |
| ZuCo2 | Shuffled pairs | 0.102% | 0.015474 |

ZuCo1 has 592 queries and 300 candidates; ZuCo2 has 983 queries and 344
candidates. Gaussian features outperform real EEG on ZuCo1. On ZuCo2 real EEG
has slightly higher MRR but fewer top-1 hits (4 versus 5). These outcomes do not
show a consistent real-EEG advantage under this adaptation. No significance
claim is made from this single split.

![Frozen LaBraM controls](labram_controls_mrr.png)

The export contains two timestamped report sets for each control, `054736` and
`060032` on 2026-10-29. All exported fields other than filenames are identical
within each pair. Each condition is displayed once; no scores are averaged or
selected for better performance. Both source filenames remain in
[labram_controls_summary.csv](labram_controls_summary.csv), and all original
rows are retained. Equality of exported values does not prove identical
checkpoint bytes or independent replication. The summarizer rejects differing
repeated-condition values pending explicit review.

## Unseen subjects and unseen sentence content

Means below give each seed (42, 43, 44) equal weight. SD is sample SD across
splits, not a confidence interval. Splits reuse participants and stimuli and
cannot be treated as independent participant cohorts.

| Dataset | Condition | Mean Recall@1 | Mean MRR ± split SD |
|---|---|---:|---:|
| ZuCo1 | Real EEG | 2.335% | 0.085268 ± 0.007368 |
| ZuCo1 | Gaussian | 2.335% | 0.086335 ± 0.009157 |
| ZuCo1 | Shuffled | 2.895% | 0.090252 ± 0.014689 |
| ZuCo2 | Real EEG | 1.337% | 0.073448 ± 0.006412 |
| ZuCo2 | Gaussian | 1.691% | 0.075441 ± 0.003512 |
| ZuCo2 | Shuffled | 1.348% | 0.068180 ± 0.011447 |

ZuCo1 uses 60 candidates (chance MRR 0.077998) and 117/111/115 queries by seed.
ZuCo2 uses 68 candidates (chance MRR 0.070648) and 197/203/193 queries. These
percentages cannot be directly compared with 300/344-candidate retrieval. Real
EEG exceeds Gaussian MRR only on seed 43 in each dataset.

![Unseen-content splits](unseen_content_mrr.png)

## Robustness and quality weighting

The ZuCo2 full-sentence ridge comparison uses one subject split, 984 queries,
344 candidates and one corruption seed. Clean MRR is 0.023036 for standard
ridge versus 0.020230 for the quality-aware variant. Quality-aware MRR is lower
in 8 of 9 nonzero corruption conditions; the exception is 25% channel dropout.
No consistent benefit is demonstrated for the implemented weighting heuristic.

Corruption curves are non-monotonic near chance. Temporal masking sometimes
increases retrieval scores, which does not establish that removing signal is
beneficial. Low baseline performance, small hit counts, and random corruption
variation limit robustness conclusions. Each corruption has different severity
units, so separate panels are used.

![Robustness comparison](quality_robustness_mrr.png)

## Comparison with published work

A source-verified [literature and protocol comparison](../published_work_comparison.md)
reviews the official AAAI 2022 EEG-to-text implementation, its acknowledged
evaluation correction and the official LaBraM task/input requirements. Primary
sources are pinned and hashed in [literature_sources.json](literature_sources.json).

Published generation BLEU/ROUGE scores and LaBraM classification results cannot
be ranked against our fixed-pool sentence Recall@K/MRR. Word-level spectral EEG,
all-subject sentence-index partitions, language decoders, montage adaptation and
different candidate pools further limit comparability. This review establishes
methodological context, not a reproduced numerical benchmark or competitive
decoding claim. Its correction discussion is attributed to the original
repository notice and the correction maintainers' documentation.

## Wider-grid exploratory cross-dataset transfer

The 18 wider-grid evaluations are **separate from the original-grid experiments**. They cover both directions, seeds 42/43/44 and real, Gaussian and shuffled statistical-ridge conditions. The original grid `[0.1, 1, 10, 100]` was run only at seed 42 for transfer. The wider grid is fixed at `[0.1, 1, 10, 100, 1000, 10000, 100000]`. Do not pool these grids or treat the original-grid runs as extra repeats.

Inputs are the duration-QC 105-channel, 200 Hz normal-reading cohorts with full recordings. These are statistical features, not the 19-channel LaBraM adaptation. Normalization, feature scaling and ridge fitting use **source training observations only**. Source-validation MRR selects alpha, with the first maximum winning ties; target scores never select alpha. Gaussian validation/evaluation uses Gaussian inputs, while shuffled training uses true validation/test labels.

Every run retains `protocol: transfer_exploratory` and `identity_audited: false`. Cross-release participant independence is **unresolved**. Different subject prefixes do not verify different people. Inclusive target pools contain 50 texts shared with the source release; these results establish neither unseen-person nor unseen-text transfer. The strict transfer identity gate remains unchanged and strict transfer is unexecuted.

| Direction | Queries per run | Target subjects | Unique candidates | Uniform-ranking MRR |
|---|---:|---:|---:|---:|
| ZuCo1 → ZuCo2 | 5,884 | 18 | 344 | 0.018661 |
| ZuCo2 → ZuCo1 | 3,349 | 12 | 300 | 0.020942 |

Target query digests and ordered candidate hashes match across all nine runs per direction. The same target observations and participants recur at every seed. Source splits overlap and noise/shuffle draws change with seed. Equal-weight means and sample SD across three seeds describe this variation; **repeated splits are not independent cohorts**. SD is not a confidence interval or standard error. Gaussian inputs retain duration and per-channel amplitude, so uniform-ranking chance cannot substitute for these matched controls.

### Seed-level wider-grid results

| Direction | Seed | Condition | Selected alpha | Source validation MRR | Target Recall@1 | Target MRR |
|---|---:|---|---:|---:|---:|---:|
| ZuCo1 → ZuCo2 | 42 | Real | 100 | 0.028249 | 0.340% | 0.018732 |
| ZuCo1 → ZuCo2 | 42 | Gaussian | 100 | 0.025700 | 0.306% | 0.019507 |
| ZuCo1 → ZuCo2 | 42 | Shuffled | 100000 | 0.020376 | 0.272% | 0.018177 |
| ZuCo1 → ZuCo2 | 43 | Real | 1000 | 0.024743 | 0.255% | 0.019005 |
| ZuCo1 → ZuCo2 | 43 | Gaussian | 0.1 | 0.027825 | 0.493% | 0.020673 |
| ZuCo1 → ZuCo2 | 43 | Shuffled | 0.1 | 0.027011 | 0.238% | 0.017791 |
| ZuCo1 → ZuCo2 | 44 | Real | 0.1 | 0.022323 | 0.238% | 0.017851 |
| ZuCo1 → ZuCo2 | 44 | Gaussian | 10 | 0.020144 | 0.238% | 0.017859 |
| ZuCo1 → ZuCo2 | 44 | Shuffled | 100 | 0.022480 | 0.238% | 0.018772 |
| ZuCo2 → ZuCo1 | 42 | Real | 1000 | 0.026690 | 0.627% | 0.024933 |
| ZuCo2 → ZuCo1 | 42 | Gaussian | 1000 | 0.024125 | 0.448% | 0.023354 |
| ZuCo2 → ZuCo1 | 42 | Shuffled | 100000 | 0.019260 | 0.299% | 0.020941 |
| ZuCo2 → ZuCo1 | 43 | Real | 1000 | 0.024824 | 0.508% | 0.024221 |
| ZuCo2 → ZuCo1 | 43 | Gaussian | 1 | 0.025861 | 0.358% | 0.022323 |
| ZuCo2 → ZuCo1 | 43 | Shuffled | 1000 | 0.020172 | 0.388% | 0.021521 |
| ZuCo2 → ZuCo1 | 44 | Real | 100000 | 0.021306 | 0.388% | 0.021621 |
| ZuCo2 → ZuCo1 | 44 | Gaussian | 10 | 0.025757 | 0.567% | 0.023748 |
| ZuCo2 → ZuCo1 | 44 | Shuffled | 1 | 0.016905 | 0.418% | 0.021185 |

![Seed-level wider-grid MRR](wider_grid_transfer/wider_grid_seed_mrr.png)

Figure: all 18 wider-grid runs. Dashed lines show uniform-ranking expectations, not significance thresholds. Full-precision metrics are in [runs.csv](wider_grid_transfer/runs.csv).

### Descriptive means and paired controls

| Direction | Condition | Mean Recall@1 ± split SD (percentage points) | Mean MRR ± split SD |
|---|---|---:|---:|
| ZuCo1 → ZuCo2 | Real | 0.278% ± 0.055 | 0.018529 ± 0.000603 |
| ZuCo1 → ZuCo2 | Gaussian | 0.346% ± 0.132 | 0.019346 ± 0.001414 |
| ZuCo1 → ZuCo2 | Shuffled | 0.249% ± 0.020 | 0.018247 ± 0.000494 |
| ZuCo2 → ZuCo1 | Real | 0.508% ± 0.119 | 0.023592 ± 0.001743 |
| ZuCo2 → ZuCo1 | Gaussian | 0.458% ± 0.105 | 0.023142 ± 0.000736 |
| ZuCo2 → ZuCo1 | Shuffled | 0.368% ± 0.062 | 0.021216 ± 0.000291 |

![Wider-grid means and SD](wider_grid_transfer/wider_grid_mean_sd.png)

Figure: bars are equal-weight means, error bars are sample SD (ddof=1), and points are individual seeds. These reuse target cohorts.

| Direction | Seed | Real − Gaussian Recall@1 (pp) | Real − Gaussian MRR |
|---|---:|---:|---:|
| ZuCo1 → ZuCo2 | 42 | +0.033990 | -0.000775 |
| ZuCo1 → ZuCo2 | 43 | -0.237933 | -0.001668 |
| ZuCo1 → ZuCo2 | 44 | +0.000000 | -0.000008 |
| ZuCo2 → ZuCo1 | 42 | +0.179158 | +0.001579 |
| ZuCo2 → ZuCo1 | 43 | +0.149298 | +0.001898 |
| ZuCo2 → ZuCo1 | 44 | -0.179158 | -0.002127 |

| Direction | Mean ΔRecall@1 ± SD (pp) | Mean ΔMRR ± SD | Positive ΔMRR seeds |
|---|---:|---:|---:|
| ZuCo1 → ZuCo2 | -0.067981 ± 0.148161 | -0.000817 ± 0.000831 | 0/3 |
| ZuCo2 → ZuCo1 | +0.049766 ± 0.198815 | +0.000450 ± 0.002237 | 2/3 |

![Paired wider-grid differences](wider_grid_transfer/wider_grid_real_minus_gaussian.png)

Differences pair real and Gaussian within direction and seed before calculating their mean and sample SD. Forward real MRR is below Gaussian on all three seeds. Reverse real MRR is higher on seeds 42 and 43 and lower on seed 44; its small positive mean difference is accompanied by substantially larger descriptive variation. **These configurations do not demonstrate reliable retrieval beyond controls.** No significance claim, pooled independent query count, or general claim about the absence of semantic information in EEG follows.

### Original grid versus wider grid: seed 42 only

| Direction | Condition | Original alpha / MRR | Wider alpha / MRR |
|---|---|---:|---:|
| ZuCo1 → ZuCo2 | Real | 100 / 0.018732 | 100 / 0.018732 |
| ZuCo1 → ZuCo2 | Gaussian | 100 / 0.019507 | 100 / 0.019507 |
| ZuCo1 → ZuCo2 | Shuffled | 1 / 0.019001 | 100000 / 0.018177 |
| ZuCo2 → ZuCo1 | Real | 10 / 0.022941 | 1000 / 0.024933 |
| ZuCo2 → ZuCo1 | Gaussian | 100 / 0.024007 | 1000 / 0.023354 |
| ZuCo2 → ZuCo1 | Shuffled | 10 / 0.021036 | 100000 / 0.020941 |

The seed-42 reverse real-EEG regularization gain remains a measured result: Recall@1 rises from 11/3,349 (0.328%) to 21/3,349 (0.627%), and MRR from 0.022941 to 0.024933. It is not a three-seed original-versus-wider ablation: the original grid was not repeated for seeds 43/44. Wider-grid selection varies by seed/control and reverse real EEG chooses the fixed upper boundary at seed 44. No additional grid search was run.

### Archive, provenance and validation

The original 79-row [retrieval.csv](retrieval.csv) is unchanged and contains neither these 18 wider-grid transfer evaluations nor the six original-grid transfer comparisons. They remain in separate archives. The [archive manifest](wider_grid_transfer/archive_manifest.json) lists source paths and SHA256 checksums. Archived artifacts include the [18-run summary](wider_grid_transfer/summary.json), [configurations and candidate pools](wider_grid_transfer/configurations_and_pools.json), per-condition statistics, paired differences, the six original-grid comparisons, copied evaluation JSONs, exact YAML recipes and implementation snapshots. Original datasets, checkpoints and experiment files are unchanged. Consolidation only reads stored metrics and renders artifacts; it launches no additional experiments.

Recorded validation: **139 passed, 2 optional skips, 0 deselected**, from `.venv/Scripts/python.exe -m pytest -q -rs`. This is the recorded pre-consolidation software suite, not a newly executed experiment or a claim that consolidation reran the suite. Both optional tests skip because `EEG_LABRAM_CACHE` is unset:

- `tests/test_storage_and_pretrained.py::test_official_pretrained_weights_on_synthetic_input`: `optional official-weight integration; no network download during unit tests`.
- `tests/test_storage_and_pretrained.py::test_pretrained_cache_extraction`: `optional official-weight extraction on synthetic fixtures`.

The formerly disk-limited `tests/test_mat5_streaming.py::test_file_above_old_limit_without_full_materialization` was included and passed. The two existing short-signal matplotlib warnings remain documented. No disk-space exclusion applies to the recorded 139-test validation.

[Consolidation artifact checks](wider_grid_transfer/consolidation_validation.json)
verify all 18 editable seed-table rows, package relationships, source checksums
and preserved experiment files. The 20 slides were rendered with Microsoft
PowerPoint and reviewed as images; a read-only text-bounds check found no
overflow. These artifact checks are separate from the recorded software suite.

## Validation-only TRT spectral and incremental-information experiments

**All scores in this section are validation-only model-selection results.** They are not test evaluations and are not appended to the original 79-report retrieval CSV or the wider-grid transfer tables. The same ZuCo1 seed-42 cohort is reused: 2,245 training observations, 512 validation queries from ZJS/ZKW and 298 candidates. Sentence classes recur across readers. There are no new exclusions. The all-missing training observation ZMG/70 is retained with training-fitted imputation. Original duration-QC exclusions remain unchanged; no test features were extracted for these pilots.

### Structural retrieval performance is a separate finding

The **count/duration structural baseline** reaches validation Recall@1/5/10 of **17.7734% / 40.6250% / 51.3672%**, with **MRR 0.292012**. It uses 14 count/duration summaries and 48 training-derived word-count basis columns. It has **no measured TRT amplitudes**. This is performance from annotation-derived reading structure, not evidence that EEG amplitudes retrieve meaning. Eight candidate sentences have unique word counts and count buckets contain at most 16 of the 298 candidates. No word text or sentence identity is supplied as a model input, but this strong structural shortcut remains.

### Spectral pilot: validation-only selected results

The new TRT view pools eight 105-channel source word vectors into 840 sentence amplitude means. The spectral arms also receive three count/missingness summaries and the training-derived word-count basis (891 features). Count/duration-only has 62 inputs. Zero/missingness-only retains structure without amplitude information. Gaussian values preserve word counts and exact missingness patterns, with marginal moments fitted on training word values only. Class shuffling consistently deranges training sentence labels while validation labels remain correct.

| Condition | Validation Recall@1 | Validation Recall@5 | Validation Recall@10 | Validation MRR |
|---|---:|---:|---:|---:|
| real_trt | 12.3047% | 32.2266% | 41.4062% | 0.226986 |
| matched_gaussian | 10.7422% | 23.6328% | 28.9062% | 0.177208 |
| class_shuffled | 0.3906% | 2.1484% | 3.7109% | 0.022676 |
| count_duration_only | 17.7734% | 40.6250% | 51.3672% | 0.292012 |
| zero_missingness_only | 16.4062% | 40.2344% | 52.3438% | 0.281028 |
| existing_sentence_ridge | 0.5859% | 2.7344% | 4.1016% | 0.026320 |

Each pilot arm selected from **28 settings**: two imputers × two scalers × seven ridge alphas `[0.1,1,10,100,1000,10000,100000]`. All imputer/scaler parameters use training data only; validation MRR selects settings. Real TRT exceeds the Gaussian and existing sentence-ridge comparators, but it trails **both structural controls on all reported recall cutoffs and MRR**. Real-minus-count/duration MRR is −0.065026; real-minus-zero/missingness MRR is −0.054043. The large gain over the sentence ridge does not establish an EEG-content advantage.

The existing sentence ridge uses its original 1,050 statistics/Welch features from normalized 200 Hz voltage and an at-most-eight-second crop, freshly fitted on the same eligible cohort. The supplied TRT features use different producer fixation windows, so this is not an alignment-only ablation. [Pilot methods and full grids](../spectral_trt_pilot_v1_20261030.md) and [archived pilot evidence](spectral_trt_pilot_v1_20261030/README.md) retain all details.

### Incremental information: identical structural inputs, validation only

All three incremental conditions use the **same 62 structural inputs and frozen training-fitted structural imputer/scaler**. The combined conditions append only 840 real or matched Gaussian amplitude means. Structural-only chooses among 28 settings. Each combined model searches the **same 196 settings**: two EEG imputers × two EEG scalers × seven alphas × seven post-scaling EEG weights `[0,0.01,0.03,0.1,0.3,1,3]`. Applying weights after scaling prevents the scaler from cancelling them. Zero EEG weight reproduces the structural-only path exactly at each alpha. Zero wins ties; all three selected alpha 10 and both combined models selected EEG weight 0.01.

| Condition | Validation Recall@1 | Validation Recall@5 | Validation Recall@10 | Validation MRR |
|---|---:|---:|---:|---:|
| structural_only | 17.7734% | 40.6250% | 51.3672% | 0.292012 |
| structural_real | 17.3828% | 41.0156% | 51.5625% | 0.292310 |
| structural_gaussian | 17.7734% | 40.8203% | 51.1719% | 0.292156 |

| Selected validation difference | ΔRecall@1 (pp) | ΔRecall@5 (pp) | ΔRecall@10 (pp) | ΔMRR |
|---|---:|---:|---:|---:|
| structural_real_minus_structural_only | -0.3906 | +0.3906 | +0.1953 | +0.000299 |
| structural_real_minus_structural_gaussian | -0.3906 | +0.1953 | +0.3906 | +0.000154 |
| structural_gaussian_minus_structural_only | +0.0000 | +0.1953 | -0.1953 | +0.000144 |

Real TRT has a **tiny selected aggregate validation MRR increase** over both models, but loses two Recall@1 hits. Reader-level differences contradict a consistent benefit:

| Validation reader | Queries | Real − structural MRR | Real − Gaussian-combined MRR |
|---|---:|---:|---:|
| zuco1:ZJS | 216 | -0.003530 | -0.004365 |
| zuco1:ZKW | 296 | +0.003092 | +0.003452 |

**These configurations do not demonstrate a consistent useful incremental EEG advantage.** The structural baseline is strong; that finding must remain separate from the absence of a consistent additional contribution from real TRT. Structural-only here reproduces the pilot count/duration baseline and is **not an independent replication**. [Incremental methods and all settings](../trt_incremental_v1_20261030.md) and [the incremental archive](trt_incremental_v1_20261030/README.md) preserve predictions and preprocessing provenance.

### Repeated validation use and stopping status

The pilot selected six models from 28 settings each. The incremental study selected structural preprocessing from 28 settings and each combined model from **196 settings**. This gives combined models more search opportunity than structural-only, while their real/Gaussian searches match. This validation split was also used by earlier CNN, pooling and spectral investigations. **Repeated validation use and maximization over settings create selection optimism.** The scores are exploratory selection results, not independent held-out estimates, significance tests or evidence from replicated cohorts. No confidence interval based on independent words or repeated uses of these readers is justified.

Source TRT values are Hilbert-envelope amplitude summaries, not squared power or waveforms. Stored duration controls use **sample counts at 500 Hz**, not milliseconds. Source gaze alignment, full-session preprocessing, the documented gamma2 edge discrepancy and sampled fixation boundary tails remain limitations. All new learned preprocessing uses training data only. Word text and sentence IDs remain outside the model-input columns; the proposed punctuation repair remains unapplied.

**Further tuning and test evaluation are stopped.** No new fitting, tuning, feature extraction or model evaluation ran during this consolidation. The original 79 reports, transfer grids, standalone studies, feature caches and checkpoints remain preserved.

Recorded targeted software checks were **7 passed** for the spectral pilot and **4 passed** for incremental ridge. They are separate targeted checks, not a new cumulative full-suite count. The historical 139-passed/two-optional-skip record elsewhere in this report remains historical. Consolidation performs only evidence/package/render checks.

[Consolidated validation evidence](spectral_validation_consolidation_20261030/stored_evidence.json) retains full precision, the common observation/candidate lists and source-report provenance.

## Completion checklist

| Requirement | Evidence/status |
|---|---|
| Modular local/Colab implementation and tests | Recorded suite: 139 passed, 2 optional skips (reasons above), 0 deselected; real Colab execution not established |
| Both datasets, lazy preparation, metadata and duration QC | Executed; original timing/reference audit remains limited |
| Statistical and supervised CNN baselines | Executed on both versions |
| ZuCo-trained SSL and semantic alignment | Executed on both versions; reconstruction improvement did not establish decoding |
| Official pretrained EEG comparison | Executed with documented approximate 19-channel adaptation |
| Noise and shuffled controls | Executed for statistical ridge, supervised CNN, full-sentence SSL/alignment and adapted LaBraM ridge on both versions; conclusions limited to these tested configurations |
| Quality-aware and corruption evaluation | Executed for ZuCo2 ridge; conclusions limited to that comparison |
| Unseen-subject evaluation | Executed; broader neural split variability remains unexecuted |
| Unseen-subject and unseen-content evaluation | Ridge/control comparison executed for three seeds on each version |
| ZuCo1 → ZuCo2 and reverse transfer | 18 wider-grid exploratory evaluations across seeds 42–44, plus six distinct original-grid seed-42 comparisons; strict participant-independent transfer remains unexecuted |
| Validation-only spectral and incremental studies | Completed on unchanged seed-42 train/validation cohort; structural baseline reported separately; no consistent incremental EEG advantage; test features unextracted |
| Published-work discussion | Primary-source protocol comparison completed; protocol-matched numerical reproduction remains unexecuted |
| Figures and consolidated analysis | Original figures retained; three new wider-grid figures show seed results, mean/SD and paired control differences |
| Final submission/presentation | Updated 26-slide review deck and PDF with transfer results, provenance and notes; original 14-slide version retained; final submission remains pending review |

The supplied ZuCo2 paper confirms 18 released participants and repeated
stimuli, but does not explicitly establish cohort independence from ZuCo1.
Different labels alone are insufficient. No invented participant identities or
strict-audit bypass is used. Local NR manifests have 50 shared canonical texts;
transfer must disclose overlap and distinguish unseen-content transfer.
The user initially chose to seek primary identity evidence. A
[maintainer inquiry](../participant_identity_request.md)
was prepared with the exact question, paper-listed contact and affected subject
labels. The user has since shown it in Gmail's Sent folder; no response has been
provided yet. No message was sent by the assistant.

The user has since chosen to continue without waiting for that reply.
[Training-subset diagnostics and explicitly exploratory transfer](../continue_work.md)
are executed locally. Both diagnostics memorized eight training examples; this
is optimization evidence only. The exploratory transfer findings are reported above.
The new `transfer_exploratory` protocol records participant independence as
unverified; it must not be described as strict unseen-person transfer. The
original strict protocol remains available with its identity requirement.

[Transfer preparation](../transfer_preparation.md) can assemble both directions,
validate shared signals/text targets and preserve the strict identity gate while
waiting. These preparation artifacts are not executed transfer results. A
[presentation outline](../presentation_outline.md) and an editable
[26-slide presentation draft](../presentation/NeuroText_consolidated.pptx) are available
for final deliverable preparation; the original 14-slide deck remains archived.

The user's subsequent preparation transcript reports 9,233 signal references,
594 unique sentence targets and a rebuilt shared text cache pinned to revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The reported plan status is
`blocked_pending_participant_identity`; this records the strict identity gate,
which remains in force. At preparation time no EEG copying or transfer training
was performed. The subsequent explicitly exploratory runs used that unchanged
local plan and shared cache. The text-cache merge disagreement was resolved by
re-encoding all targets together while preserving the original caches/results.

## Reproducibility and limits of the export

All 79 rows are preserved: the original 63, four added CNN controls, eight
LaBraM control reports (two timestamped reports per dataset/control condition),
and four full-sentence SSL controls.
Eight groups have identical exported values,
listed in [export_audit.json](export_audit.json); this does not prove checkpoint
identity. Corruption conditions, zero-severity evaluations, reruns and the
superseded numerically unstable ridge fit are not independent methods or
replications. The CSV lacks original artifact/config/split identities, candidate
counts, per-subject statistics and confidence intervals. Method-specific cohort
and candidate counts above also use the previously supplied execution logs.
Archive the original JSON files, split manifests and checkpoints before final
submission. Do not infer participant-cluster intervals from aggregate CSV rows.

Rebuild generated tables and figures from the repository root:

```bash
python scripts/summarize_research.py
```

The script validates required metric ranges, exactly three seeds per unseen-
content condition and unique corruption severity points. Outputs include PNG/SVG
figures, [unseen_content_summary.csv](unseen_content_summary.csv), and
[cnn_controls_summary.csv](cnn_controls_summary.csv), and
[labram_controls_summary.csv](labram_controls_summary.csv), and
[ssl_controls_summary.csv](ssl_controls_summary.csv). The CNN, SSL and LaBraM
summaries also check within-dataset protocol, seed, candidate hash, query count
and control labels. Repeated LaBraM reports are summarized once only if every
exported field except filenames agrees.
