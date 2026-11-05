# Concise research review — 2026-11-01

**The tested configurations do not demonstrate a useful, consistent EEG contribution beyond the strongest matched controls.** This does not show that EEG contains no language information. This review checks saved artifacts only; it launches no experiment or evaluation.

The [updated main report](../REPORT.md) contains the complete scope-separated tables, method inventory, control differences and selection history. [Supervisor note](SUPERVISOR_NOTE.md). [Byte-identical pre-update report/presentation archive](archive_manifest.json). The presentation remains unchanged and does not include the later pilots.

## Changes, controls and outcomes

| Study | Change and matched checks | Supported finding |
|---|---|---|
| Waveform/statistical/CNN/SSL/LaBraM | Learned or frozen representations; Gaussian and shuffled controls | No consistent useful EEG advantage across the matched comparisons. |
| CNN candidate loss / temporal pooling | Full training candidates or source-time bins; real/Gaussian/shuffled fits | Loss change does not improve real top-1; pooling does not beat its Gaussian control. |
| Spectral / incremental TRT | Word amplitude means, then a fixed structural block plus real or Gaussian EEG | Structure is strong; adding EEG gives tiny selected MRR changes and inconsistent reader effects. |
| Ordered-word ridge | PCA position bins versus averaging; Gaussian, order/class shuffle, mask and structure | Small mean-to-order difference does not exceed stronger controls. |
| Relative-amplitude/head audit | Full-width relative amplitudes; semantic versus known-class targets | Both combined heads select zero EEG weight. |
| GRU / full-candidate GRU loss | Full ordered sequences, then loss-only change; matched initialization and controls | Prediction concentration falls with full-candidate loss, but retrieval does not improve. |
| Word pretraining | Exact exported labels as targets; projection transfer, fresh baselines and matched two-stage controls | No useful sentence gain over fresh training or strongest controls. |

## Training-reader CV

Selected equal-reader development scores from the original eight training readers; later word pretraining uses a fixed final checkpoint. Recall is percent. **MRR is not accuracy.** These are not outer-validation or test results. Complete conditions: [training_cv.csv](training_cv.csv).

| Study / condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| amplitude/semantic_embedding/eeg_only | 0.3547 | 1.3798 | 2.7290 | 0.020519 |
| amplitude/semantic_embedding/gaussian_only | 0.6959 | 2.3128 | 4.2199 | 0.025876 |
| amplitude/semantic_embedding/structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| amplitude/known_sentence_class/structural_only | 20.9088 | 63.4175 | 89.3180 | 0.400506 |
| gru/real_eeg | 0.5424 | 1.7590 | 3.4716 | 0.023048 |
| gru/matched_gaussian | 0.4043 | 1.6910 | 3.5333 | 0.021327 |
| gru_ce/real_eeg | 0.5293 | 1.2973 | 2.6902 | 0.020781 |
| gru_ce/missingness_only | 0.4490 | 1.9321 | 3.9432 | 0.023184 |
| word_pretrain/pretrained_real | 0.3946 | 1.3816 | 2.9861 | 0.020149 |
| word_pretrain/fresh_same_sentence | 0.5294 | 1.5535 | 2.9032 | 0.021318 |
| word_pretrain/fresh_equal_steps | 0.3485 | 1.5793 | 2.8899 | 0.019855 |
| word_pretrain/pretrained_gaussian | 0.2160 | 1.0232 | 2.2952 | 0.017447 |
| word_pretrain/pretrained_missingness | 0.8977 | 2.8463 | 5.0861 | 0.030513 |

## Outer validation

Previously executed results on 512 reused validation queries and 298 candidates. No new outer evaluation occurred. The later amplitude/GRU/loss/word-pretraining pilots have no outer results. Complete conditions: [outer_validation.csv](outer_validation.csv).

| Study / condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| spectral/real_trt | 12.3047 | 32.2266 | 41.4062 | 0.226986 |
| spectral/matched_gaussian | 10.7422 | 23.6328 | 28.9062 | 0.177208 |
| spectral/count_duration_only | 17.7734 | 40.6250 | 51.3672 | 0.292012 |
| incremental/structural_real | 17.3828 | 41.0156 | 51.5625 | 0.292310 |
| incremental/structural_only | 17.7734 | 40.6250 | 51.3672 | 0.292012 |
| incremental/structural_gaussian | 17.7734 | 40.8203 | 51.1719 | 0.292156 |
| word_order/ordered_real | 0.3906 | 1.3672 | 4.1016 | 0.022452 |
| word_order/matched_gaussian | 0.7812 | 1.9531 | 4.1016 | 0.026921 |
| word_order/structural_only | 17.5781 | 40.4297 | 51.3672 | 0.291369 |

## Historical test

Selected old test comparisons below retain their original cohorts and candidate pools; no test inference ran. The full report includes SSL, CNN-loss, unseen-content and other historical conditions. Different cohorts/pools prevent one final-accuracy ranking. The separately stored ZJS_validation file is a manifest audit without retrieval metrics, excluded from performance tables.

| Historical method / dataset / condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| CNN/zuco1/real | 0.3378 | 1.8581 | 3.2095 | 0.020596 |
| CNN/zuco1/gaussian | 0.3378 | 1.8581 | 3.8851 | 0.022081 |
| CNN/zuco2/real | 0.3049 | 0.8130 | 2.5407 | 0.016582 |
| CNN/zuco2/gaussian | 0.3049 | 1.5244 | 3.2520 | 0.019452 |
| LaBraM/zuco1/real | 0.1689 | 2.1959 | 4.3919 | 0.022326 |
| LaBraM/zuco1/gaussian | 0.6757 | 2.3649 | 4.7297 | 0.028627 |
| LaBraM/zuco2/real | 0.4069 | 2.2380 | 3.9674 | 0.023559 |
| LaBraM/zuco2/gaussian | 0.5086 | 1.9329 | 3.2553 | 0.022214 |

Cross-release transfer is a separate exploratory scope, not an independent target-test cohort. Forward real-minus-Gaussian MRR is negative across the saved seeds; reverse differences change sign. Participant overlap between releases remains unresolved. [Transfer metrics](exploratory_transfer.csv).

## Controls, structure and reader consistency

Strong structural retrieval uses annotation-derived count/duration information; it does not establish incremental EEG information. Both amplitude combined heads select zero EEG weight. Incremental validation gains in aggregate MRR have opposite signs for the two readers and lower Recall@1.

| Real condition minus comparator | ΔR@1 pp | ΔMRR | Readers with positive R@1 gap |
|---|---:|---:|---:|
| gru - matched_gaussian | +0.1382 | +0.001721 | 3/8 |
| gru - structural_only | -15.5705 | -0.257222 | 0/8 |
| gru_ce - matched_gaussian | +0.2688 | +0.002294 | 3/8 |
| word_pretrain - fresh_same_sentence | -0.1349 | -0.001169 | 0/8 |
| word_pretrain - fresh_equal_steps | +0.0461 | +0.000294 | 3/8 |
| word_pretrain - pretrained_gaussian | +0.1785 | +0.002702 | 4/8 |
| word_pretrain - pretrained_missingness | -0.5031 | -0.010364 | 0/8 |

Full Recall@1/5/10 and MRR differences, including historical controls and individual readers: [control_differences.json](control_differences.json), [participant_metrics.csv](participant_metrics.csv). Positive counts are descriptive, not significance tests.

## Word task, separately

Fold-training exact-string classes; text is target metadata only. Missing words remain, and out-of-vocabulary targets count as misses. These are word scores, not sentence accuracy. Reader folds repeat sentence classes; no unseen-word claim follows.

| Word-stage condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| pretrained_class_shuffled | 0.0668 | 0.1988 | 0.3240 | 0.003046 |
| pretrained_gaussian | 3.0019 | 9.7271 | 15.5035 | 0.067224 |
| pretrained_missingness | 3.3331 | 14.5043 | 21.8359 | 0.088084 |
| pretrained_real | 3.0075 | 10.0459 | 15.9434 | 0.068325 |

## Interpretation and remaining uncertainty

Spectral/incremental settings were selected on reused outer validation. Ordered ridge and amplitude settings used training-reader folds; both GRU objectives selected epoch 8 from 4/8/12 on real-EEG CV. Word pretraining fixed its stages and supplied both same-sentence and equal-update fresh baselines; optimizer-update matching is not FLOP matching. Repeated development use, overlapping folds and repeated words/sentences limit every claim. These are not independent cohorts.

No confirmed current pairing/ranking bug explains the negative scores. Earlier source/cache and saved-rank checks passed. Exported-text versus stimulus-CSV differences and execution-environment problems do not establish EEG mispairing. Amplitude differences, retained PCA variance and prediction concentration are observations, not proven causes. The tested follow-ups did not establish the proposed improvement.

Source gaze-assisted segmentation, upstream session processing and physical boundary uncertainties remain. Gaussian controls are limited null models: historical waveform nulls retain observation-specific amplitude, while TRT nulls preserve masks/counts with training-word moments. Better independent reader/stimulus data are needed for generalization claims. Any future experiment needs a revised falsifiable hypothesis or narrower question, not another broad search on this development history. Work stops here.

[Verification](verification.json) reports saved-result arithmetic and artifact checks, not a new model reproduction. [Preservation](preservation_after.json) confirms prior files and dataset metadata, with only the authorized main-report edit and this newly created builder changing during consolidation.
