# Preliminary ZuCo1 results

These measurements were reported from the user's Windows RTX 3050 execution and
reviewed on 2026-10-28. The exported terminal table was supplied to this session;
the original result JSON files, signal caches and checkpoints remain on that
machine. The values below have not been independently rerun in the cloud.

## Original cohort: 3,356 observations, 12 subjects

The seed-42 held-out-subject evaluations contain 592 queries and 300 unique
candidate sentences. Neural and statistical inputs use a deterministic center
crop of at most 1,600 samples (8 seconds at 200 Hz).

| Condition | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR |
|---|---:|---:|---:|---:|
| Stable float64/SVD ridge, real EEG | 0.676 | 2.365 | 3.885 | 0.025930 |
| Stable ridge, Gaussian training and evaluation inputs | 0.507 | 2.534 | 3.885 | 0.024614 |
| Stable ridge, shuffled training pairs, real evaluation inputs | 0.507 | 1.182 | 2.534 | 0.019832 |
| Supervised CNN | 0.338 | 1.858 | 3.209 | 0.020596 |
| SSL-initialized CNN alignment, batch size 2 | 0.338 | 1.520 | 2.534 | 0.020596 |
| Uniform random-ranking expectation | 0.333 | 1.667 | 3.333 | 0.020942 |

Use `zuco1_ridge_stable.json` for the corrected ridge fit. The original
`zuco1_ridge.json` fit emitted a conditioning warning and is superseded, not an
additional method. Both Gaussian export rows have identical metrics, as do both
shuffled rows and both supervised CNN rows. Preserve these files but verify
checkpoint identities before calling them duplicates or counting repeat runs;
identical numbers do not establish independent replication.

## Filtered cohort with complete recordings

Seven observations of at most 0.1 seconds were excluded with a saved audit trail,
leaving 3,349 observations. Source-file inspection confirmed these observations
were already short in the MATLAB exports and their texts matched the imports.
This does not verify their original trigger timing or explain the source issue.
Training on complete recordings removes the center crop. In the original cohort,
1,133 observations (33.8%) exceeded eight seconds.

| Condition | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR |
|---|---:|---:|---:|---:|
| Full-sentence SSL-initialized alignment, batch size 8 | 0.169 | 1.689 | 3.041 | 0.020680 |

This evaluation also has 592 queries and the same 300-candidate hash as the
original evaluations. Matching candidate pools do not establish matching query
identities, training cohorts, or preprocessing. Do not attribute differences
solely to cropping: cohort selection and batch size also changed.

Full-sentence SSL reconstruction reached its best validation loss of 0.356537
at zero-based epoch 44. Subsequent alignment selected epoch 0; training loss
fell from 2.085270 to 1.231229 while validation loss rose from 2.089840 to
2.209301. This is consistent with overfitting. Reconstruction improvement did
not establish successful sentence retrieval.

## Interpretation and outstanding work

These exploratory results do not establish reliable EEG-to-sentence decoding.
The real ridge result is numerically above uniform chance but close to its
Gaussian control. No statistical significance claim is supported by this table.
There is only one split/seed, with two test subjects. Observations from the same
subjects and sentences are dependent; use subject-level reporting and justified
uncertainty estimates. Successive test inspections make this an exploratory
development history, not an untouched confirmatory test campaign.

Still unexecuted in the supplied results: real-data pretrained EEG features,
ZuCo2 baselines, both transfer directions, held-out-content retrieval, matched
quality-aware comparisons, robustness curves, and repeated subject splits.
Controls for ridge do not replace controls for neural methods. Final controlled
comparisons must share audited cohorts, signal recipes, split identities and
candidate pools. Audit channel mapping and cross-version participant overlap
before pretrained extraction or transfer. Assemble final figures and discussion
after recording executed and unexecuted requirements explicitly.

## Subsequent matched pretrained comparisons (2026-10-29)

The user subsequently completed real-data frozen official LaBraM extraction and
ridge evaluation on both versions. All values below are user-reported Windows
GPU results, not cloud reruns. Inputs select the same 19 source electrodes using
the template-based approximate mapping documented in `pretrained_models.md`.
Both methods use the same prepared signals, complete seconds, seed 42, source
preprocessing, candidate pool and within-version subject split. Signals shorter
than one complete second are excluded from both methods. Mapping and upstream
preprocessing adaptation limit conclusions about the pretrained model generally.

| Dataset | Representation + ridge | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR |
|---|---|---:|---:|---:|---:|
| ZuCo1 | Statistical/spectral | 0.507 | 1.689 | 3.716 | 0.024530 |
| ZuCo1 | Frozen LaBraM | 0.169 | 2.196 | 4.392 | 0.022326 |
| ZuCo2 | Statistical/spectral | 0.814 | 2.442 | 4.374 | 0.027599 |
| ZuCo2 | Frozen LaBraM | 0.407 | 2.238 | 3.967 | 0.023559 |

ZuCo1: 3,345 prepared observations, 592 test queries, 300 candidates. Statistical
ridge has 3 top-1 hits versus 1 for LaBraM. Outputs:
`zuco1_statistical19_20261029_022627.json` and
`zuco1_labram19_20261029_022627.json`.
ZuCo2: 5,838 prepared observations, 983 test queries, 344 candidates. Statistical
ridge has 8 top-1 hits versus 4 for LaBraM. Outputs:
`zuco2_statistical19_20261029_020526.json` and
`zuco2_labram19_20261029_020526.json`.

Frozen LaBraM did not improve top-1 retrieval or MRR over the matched statistical
baseline on either reported split. ZuCo1 Recall@5/10 improved, so the outcome is
metric-dependent. No statistical significance or general superiority claim is
made. Matched negative controls and repeated splits for these adapted cohorts
remain unexecuted. The earlier 105-channel runs have different cohorts and must
not be counted as matched pretraining ablations.

Cross-version audit reported equal ordered channel layouts, 200 Hz, `nr` tasks,
and 50 shared sentence hashes in the duration-filtered manifests. Neither
manifest contains audited `person_id`. Different version-specific subject labels
alone do not establish participant independence. On 2026-10-29 the official
ZuCo2 OSF home wiki was readable at
https://api.osf.io/v2/wikis/kw2df/content/ ; it documents intentionally repeated
stimuli but does not state whether the participants overlap. Paper access from
this environment remains blocked. Transfer remains unexecuted pending identity
evidence; no artificial identity reconciliation or audit bypass is performed.
