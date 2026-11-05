# Ordered word-level TRT pilot — validation only

**No useful improvement beyond controls; stopped without test evaluation.** Ordered real Recall@1 is 0.3906%, versus 0.1953% for the matched mean baseline, 0.7812% for Gaussian and 17.5781% for structure. The mean-to-ordered difference is one extra correct validation query, not evidence of success.

This pilot tests whether retaining source word order improves over averaging word EEG. It is a small linear pilot, not a newly trained Transformer or a claimed decoding success.

## Fixed protocol and selection

Same ZuCo1 seed-42 cohort: 2,245 training observations from eight readers; 512 reused validation queries from ZJS/ZKW, with the same 298 sentence candidates. No target correction, added observation or new exclusion. No test-source files, test features or test scores are used. The two validation readers are not independent replicated cohorts.

TRT supplies eight bands × 105 ordered source channels per word. The bounded reader checks parent sentence pairing and skips word text, raw eye-tracking traces, pupil and gaze coordinates. Word identity, sentence ID and subject ID never enter numeric model matrices. The word axis is source stimulus order, not fixation chronology or inferred physical timing.

For each held-out training reader, standardization uses finite words from the remaining seven readers only. A deterministic sample of at most 8,192 of these words fits 16-component randomized PCA without whitening. Four disjoint word-position bins each average available projected words; empty bins use zero. Four valid-word fractions are appended (68 features). Mean-real uses the same projection and a single bin (17 features). A sentence-level median imputer and standard scaler fit training rows only. These are feature widths, not matched model capacities; the same-width word-order-shuffled control is needed to assess order.

The historical spectral mean baseline used 891 inputs including explicit structure and no word-level PCA. Its 12.3047% selected validation Recall@1 is preserved, but is not a matched baseline for these 17/68-input models. The earlier 17.7734% structural score was selected on outer validation MRR; the new structural setting is selected on inner-training Recall@1. Differences from those historical scores must not be attributed solely to word order. Failure of this bounded PCA/pooling pilot does not establish that every word-sequence model fails.

Each arm selects one of three ridge alphas [1, 10, 100] using **equal-reader mean Recall@1 over all eight leave-one-training-reader-out folds**. Ties prefer stronger regularization. All Gaussian moments, PCA fits, scaling and count bases are refitted inside each fold. Candidate sentence embeddings are targets/public candidates, never EEG inputs. After selection is saved, preprocessing/model parameters refit on the original eight training readers and each frozen arm is evaluated once on the outer validation cohort. Outer validation does not select alphas, epochs or transforms. This cohort has already informed earlier studies, so this remains exploratory and cannot be described as fresh independent confirmation.

## Controls and leakage limitations

Gaussian words preserve exact word counts and finite/absent positions, using training-word column means and SDs; moments and Gaussian PCA are independently fitted per training fold. Word-order shuffling permutes complete EEG vectors among complete slots within each sentence, retaining mask positions and the mean. Class shuffling deranges training sentence labels consistently across repeated observations. Missingness-only uses four bin fractions, word count, overall valid fraction and a training-derived count basis. Structural-only reuses the prior 14 count/duration columns and rebuilds its count basis within each fold. Both are non-amplitude controls. Gaussian and real arms have identical widths and alpha budgets.

All word alignment remains gaze/stimulus assisted; fractions and bin boundaries can reveal reading structure and word-count information. These results must not be called annotation-free EEG decoding. Duration values are source sample counts at 500 Hz and enter only structural controls. The prior source audit's channel/band definitions, gamma2 code/descriptor discrepancy, unverified exact producer revision, source-session preprocessing and boundary limitations remain: [feature feasibility](../../word_eeg_feasibility_20261030.md).

## Coverage

| Split | Observations | Word slots | Complete EEG words | All-missing observations |
|---|---:|---:|---:|---:|
| train | 2245 | 48865 | 32848 | 1 |
| val | 512 | 11203 | 7101 | 0 |

All-missing training observation ZMG/70 is retained; no validation observation is removed.

## Frozen selected validation scores

Recall is percent. MRR is not accuracy.

| Condition | Alpha (training CV) | Features | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|---:|---:|
| mean_real | 100 | 17 | 0.1953 | 1.7578 | 3.3203 | 0.019783 |
| ordered_real | 100 | 68 | 0.3906 | 1.3672 | 4.1016 | 0.022452 |
| matched_gaussian | 100 | 68 | 0.7812 | 1.9531 | 4.1016 | 0.026921 |
| word_order_shuffled | 1 | 68 | 0.5859 | 1.1719 | 4.6875 | 0.024464 |
| class_shuffled | 10 | 68 | 0.0000 | 2.1484 | 3.5156 | 0.019378 |
| missingness_only | 1 | 54 | 17.5781 | 40.6250 | 51.7578 | 0.291229 |
| structural_only | 1 | 62 | 17.5781 | 40.4297 | 51.3672 | 0.291369 |
| Uniform random ranking | — | — | 0.3356 | 1.6779 | 3.3557 | 0.021060 |

## Ordered real minus comparators

| Comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |
|---|---:|---:|---:|---:|
| ordered_real_minus_mean_real | +0.1953 | -0.3906 | +0.7812 | +0.002669 |
| ordered_real_minus_matched_gaussian | -0.3906 | -0.5859 | +0.0000 | -0.004469 |
| ordered_real_minus_word_order_shuffled | -0.1953 | +0.1953 | -0.5859 | -0.002012 |
| ordered_real_minus_class_shuffled | +0.3906 | -0.7812 | +0.5859 | +0.003074 |
| ordered_real_minus_missingness_only | -17.1875 | -39.2578 | -47.6562 | -0.268777 |
| ordered_real_minus_structural_only | -17.1875 | -39.0625 | -47.2656 | -0.268917 |

## Reader-level results

| Condition | Validation reader | Queries | R@1 % | MRR |
|---|---|---:|---:|---:|
| mean_real | ZJS | 216 | 0.0000 | 0.016285 |
| mean_real | ZKW | 296 | 0.3378 | 0.022335 |
| ordered_real | ZJS | 216 | 0.0000 | 0.015986 |
| ordered_real | ZKW | 296 | 0.6757 | 0.027171 |
| matched_gaussian | ZJS | 216 | 0.4630 | 0.020927 |
| matched_gaussian | ZKW | 296 | 1.0135 | 0.031295 |
| word_order_shuffled | ZJS | 216 | 0.0000 | 0.018063 |
| word_order_shuffled | ZKW | 296 | 1.0135 | 0.029136 |
| class_shuffled | ZJS | 216 | 0.0000 | 0.016383 |
| class_shuffled | ZKW | 296 | 0.0000 | 0.021565 |
| missingness_only | ZJS | 216 | 17.5926 | 0.288177 |
| missingness_only | ZKW | 296 | 17.5676 | 0.293456 |
| structural_only | ZJS | 216 | 17.5926 | 0.290040 |
| structural_only | ZKW | 296 | 17.5676 | 0.292338 |

## Prespecified descriptive screen

Before fitting, the screen required ordered real to exceed the mean baseline and every control on aggregate Recall@1 and MRR, and to exceed Gaussian and structural Recall@1 for each validation reader. This is a descriptive stopping rule, not a significance test.

Screen passed: **False**.

- `mean_real_aggregate`: True
- `matched_gaussian_aggregate`: False
- `word_order_shuffled_aggregate`: False
- `class_shuffled_aggregate`: True
- `missingness_only_aggregate`: False
- `structural_only_aggregate`: False
- `ZJS_beats_matched_gaussian`: False
- `ZJS_beats_structural_only`: False
- `ZKW_beats_matched_gaussian`: False
- `ZKW_beats_structural_only`: False

**Stop after this validation-only pilot. No additional search or test evaluation.** A higher single score cannot establish robust EEG benefit or predict future accuracy.

## Preservation and software verification

All 1109 protected prior artifacts match pre-run SHA256 hashes. Existing data, checkpoints, previous results, report and presentation are preserved. Eleven focused tests passed before the run (word-order and existing spectral-pilot tests). This is not a new full-suite test count.

Full fold scores, train-reader provenance, frozen selections, preprocessing, coefficients, candidate pools, per-query ranks and source hashes remain in `results/word_order_pilot_v1_20261031`. JSON evidence and source snapshots are archived here.
