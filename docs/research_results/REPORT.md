# EEG research review: completed experiments and evidence

Updated 2026-11-05 from saved results only. **The tested configurations do not demonstrate a useful, consistent EEG contribution beyond the strongest matched controls.** This does not establish that EEG contains no language information.

No training, tuning, feature extraction, checkpoint inference, outer-validation evaluation or test evaluation ran during this consolidation. Numerical checks summarize previously saved ranks and metrics. The [pre-update report and presentation archive](evidence_review_v1_20261101/archive_manifest.json) contains byte-identical copies. The earlier 26-slide PowerPoint/PDF remain unchanged. The [35-slide v2 deck](../presentation/NeuroText_consolidated_v2_20261105.pptx) appends the later studies and N-way analysis from saved results ([update manifest](presentation_update_20261105/manifest.json)).

## What changed and what was measured

| Experiment | Change and controls | Evidence-supported outcome |
|---|---|---|
| Historical statistical/CNN/SSL/LaBraM | Waveform statistics, supervised convolution, reconstruction pretraining and frozen adapted pretrained features; real, Gaussian and class-shuffled supervision controls | No consistent useful real-EEG advantage. Memorizing a small training subset verifies optimization capacity, not reader generalization. |
| CNN candidate loss | Fixed CNN and cohort; batch candidates versus full training-candidate loss, each with real/Gaussian/shuffled arms | Both real arms retain the same Recall@1; broader candidates do not establish a useful gain. Separate historical test comparison. |
| Temporal pooling | Global versus four source-time segments; matched real/Gaussian/shuffled fits | Same real Recall@1; the segment variant trails its Gaussian control. Validation only. |
| Spectral TRT mean | Supplied word-band amplitude means with count/mask structure; Gaussian, shuffled, structure-only, mask-only and waveform-ridge comparators | Better than a waveform baseline is not an isolated EEG increment: both non-amplitude controls are stronger. |
| Incremental TRT | Same structural block plus real or Gaussian amplitudes; post-scaling EEG weights include zero | Tiny selected MRR increment, lower Recall@1, opposite reader effects. |
| Ordered-word ridge | Training-fitted PCA and four source-word-position bins versus mean; Gaussian, order-shuffled, class-shuffled, mask and structure controls | One extra outer-validation hit versus mean; Gaussian, order-shuffled and structure remain stronger. |
| Relative-amplitude/head audit | Full-width signed-log relative channel features averaged by sentence; semantic embedding versus known-class head; Gaussian, shuffled, structure and combined arms | Structure dominates; selected combined models use zero EEG weight for both heads. |
| Full-sequence GRU | Small projection, one GRU layer, embedding head; mean, Gaussian, order-shuffled, class-shuffled, mask and structure controls | Small selected advantage over Gaussian, inconsistent across readers and far below structure; training-only gate fails. |
| GRU retrieval loss | Same inputs/architecture/budget/init; positive-only cosine versus full-candidate cross entropy | Prediction concentration decreases, but sentence retrieval does not improve. |
| Word pretraining | Exact exported-word class targets, projection-only transfer; fresh same-sentence and equal-update baselines, Gaussian/shuffled/mask stages and structure | No useful sentence-retrieval gain over fresh training or strongest controls. Word scores are reported separately. |

## Training-reader CV: sentence outcomes

These are equal-reader macro development scores on the original eight training readers and 2,245 observations, with 300 fixed sentence candidates. Recall values are percentages; **MRR is not accuracy**. Heads and representations are distinct tasks/settings. These are not outer-validation or historical test scores.

| Study / condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| gru/real_eeg | 0.5424 | 1.7590 | 3.4716 | 0.023048 |
| gru/mean_eeg | 0.4011 | 1.7373 | 3.5249 | 0.022454 |
| gru/matched_gaussian | 0.4043 | 1.6910 | 3.5333 | 0.021327 |
| gru/class_shuffled | 0.3578 | 1.8316 | 3.3172 | 0.021432 |
| gru/word_order_shuffled | 0.4412 | 1.7503 | 3.5206 | 0.022068 |
| gru/missingness_only | 0.3578 | 1.6588 | 3.3200 | 0.021191 |
| gru/structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| gru_ce/real_eeg | 0.5293 | 1.2973 | 2.6902 | 0.020781 |
| gru_ce/mean_eeg | 0.3630 | 1.4867 | 2.9624 | 0.019303 |
| gru_ce/matched_gaussian | 0.2605 | 1.2261 | 2.7938 | 0.018486 |
| gru_ce/class_shuffled | 0.1744 | 1.5069 | 3.7394 | 0.020120 |
| gru_ce/word_order_shuffled | 0.2650 | 1.3330 | 2.6156 | 0.018809 |
| gru_ce/missingness_only | 0.4490 | 1.9321 | 3.9432 | 0.023184 |
| gru_ce/structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| word_pretrain/pretrained_real | 0.3946 | 1.3816 | 2.9861 | 0.020149 |
| word_pretrain/fresh_equal_steps | 0.3485 | 1.5793 | 2.8899 | 0.019855 |
| word_pretrain/fresh_same_sentence | 0.5294 | 1.5535 | 2.9032 | 0.021318 |
| word_pretrain/pretrained_gaussian | 0.2160 | 1.0232 | 2.2952 | 0.017447 |
| word_pretrain/pretrained_class_shuffled | 0.5783 | 1.8241 | 3.7205 | 0.023627 |
| word_pretrain/pretrained_missingness | 0.8977 | 2.8463 | 5.0861 | 0.030513 |
| word_pretrain/structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| amplitude/semantic_embedding/structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| amplitude/semantic_embedding/eeg_only | 0.3547 | 1.3798 | 2.7290 | 0.020519 |
| amplitude/semantic_embedding/gaussian_only | 0.6959 | 2.3128 | 4.2199 | 0.025876 |
| amplitude/semantic_embedding/structural_real | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| amplitude/semantic_embedding/structural_gaussian | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| amplitude/semantic_embedding/structural_shuffled | 0.3603 | 2.1483 | 3.8220 | 0.021886 |
| amplitude/known_sentence_class/structural_only | 20.9088 | 63.4175 | 89.3180 | 0.400506 |
| amplitude/known_sentence_class/eeg_only | 0.3575 | 1.1614 | 2.6305 | 0.019292 |
| amplitude/known_sentence_class/gaussian_only | 0.3954 | 1.7278 | 3.8723 | 0.022132 |
| amplitude/known_sentence_class/structural_real | 20.9088 | 63.4175 | 89.3180 | 0.400506 |
| amplitude/known_sentence_class/structural_gaussian | 20.9088 | 63.4175 | 89.3180 | 0.400506 |
| amplitude/known_sentence_class/structural_shuffled | 0.5866 | 1.7994 | 3.0579 | 0.021740 |
| word_order/mean_real | 0.2149 | 1.7433 | 3.8854 | 0.020972 |
| word_order/ordered_real | 0.3708 | 2.2268 | 3.8451 | 0.023267 |
| word_order/matched_gaussian | 0.3226 | 2.4099 | 4.0192 | 0.023465 |
| word_order/word_order_shuffled | 0.5716 | 2.1317 | 3.7624 | 0.024375 |
| word_order/class_shuffled | 0.4497 | 1.4912 | 3.3873 | 0.021144 |
| word_order/missingness_only | 15.1968 | 41.1365 | 52.3709 | 0.278940 |
| word_order/structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |

The amplitude known_sentence_class head is closed-set classification of repeated known sentence classes; its structural performance is not semantic EEG decoding. Repeated structural rows reuse the same observations and recipe, not independent replications.

## Outer validation: separate reused development cohort

These previously executed scores use 512 queries from ZJS/ZKW and 298 candidates. Spectral/incremental settings were selected on this validation set. Ordered-word ridge selected alphas within training readers before its one outer evaluation, but the outer readers already had extensive development use. The later amplitude, GRU, GRU-loss and word-pretraining pilots did not evaluate outer validation.

| Study / condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| spectral/real_trt | 12.3047 | 32.2266 | 41.4062 | 0.226986 |
| spectral/matched_gaussian | 10.7422 | 23.6328 | 28.9062 | 0.177208 |
| spectral/class_shuffled | 0.3906 | 2.1484 | 3.7109 | 0.022676 |
| spectral/count_duration_only | 17.7734 | 40.6250 | 51.3672 | 0.292012 |
| spectral/zero_missingness_only | 16.4062 | 40.2344 | 52.3438 | 0.281028 |
| spectral/existing_sentence_ridge | 0.5859 | 2.7344 | 4.1016 | 0.026320 |
| incremental/structural_only | 17.7734 | 40.6250 | 51.3672 | 0.292012 |
| incremental/structural_real | 17.3828 | 41.0156 | 51.5625 | 0.292310 |
| incremental/structural_gaussian | 17.7734 | 40.8203 | 51.1719 | 0.292156 |
| word_order/mean_real | 0.1953 | 1.7578 | 3.3203 | 0.019783 |
| word_order/ordered_real | 0.3906 | 1.3672 | 4.1016 | 0.022452 |
| word_order/matched_gaussian | 0.7812 | 1.9531 | 4.1016 | 0.026921 |
| word_order/word_order_shuffled | 0.5859 | 1.1719 | 4.6875 | 0.024464 |
| word_order/class_shuffled | 0.0000 | 2.1484 | 3.5156 | 0.019378 |
| word_order/missingness_only | 17.5781 | 40.6250 | 51.7578 | 0.291229 |
| word_order/structural_only | 17.5781 | 40.4297 | 51.3672 | 0.291369 |
| pooling/global/real | 0.1953 | 1.5625 | 3.1250 | 0.018858 |
| pooling/global/gaussian | 0.1953 | 0.9766 | 2.9297 | 0.018387 |
| pooling/global/shuffled | 0.1953 | 2.3438 | 3.5156 | 0.020681 |
| pooling/four_segment/real | 0.1953 | 1.5625 | 2.7344 | 0.019086 |
| pooling/four_segment/gaussian | 0.3906 | 1.5625 | 2.5391 | 0.021347 |
| pooling/four_segment/shuffled | 0.1953 | 1.1719 | 3.1250 | 0.018683 |

## Historical test: previously saved scores only

CNN/SSL use 592 ZuCo1 queries and 984 ZuCo2 queries, with 300/344 candidates. LaBraM uses 592/983 queries and 300/344 candidates on a separately adapted channel/cohort recipe. Unseen-content rows below are equal-seed means across overlapping splits with only 60/68 candidates; their scores are not comparable to full-pool retrieval. The separate CNN-candidate comparison uses 592 queries and 300 candidates. None of these old test sets is a new independent confirmation or a current model-selection target.

| Historical method / dataset / condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| CNN/zuco1/real | 0.3378 | 1.8581 | 3.2095 | 0.020596 |
| CNN/zuco1/gaussian | 0.3378 | 1.8581 | 3.8851 | 0.022081 |
| CNN/zuco1/shuffled | 0.1689 | 1.6892 | 3.7162 | 0.020392 |
| CNN/zuco2/real | 0.3049 | 0.8130 | 2.5407 | 0.016582 |
| CNN/zuco2/gaussian | 0.3049 | 1.5244 | 3.2520 | 0.019452 |
| CNN/zuco2/shuffled | 0.2033 | 0.9146 | 2.1341 | 0.016242 |
| SSL/zuco1/real | 0.1689 | 1.6892 | 3.0405 | 0.020680 |
| SSL/zuco1/gaussian | 0.3378 | 1.6892 | 3.5473 | 0.023159 |
| SSL/zuco1/shuffled | 0.3378 | 1.8581 | 2.5338 | 0.019842 |
| SSL/zuco2/real | 0.3049 | 1.5244 | 3.2520 | 0.018846 |
| SSL/zuco2/gaussian | 0.3049 | 1.4228 | 3.1504 | 0.018117 |
| SSL/zuco2/shuffled | 0.3049 | 1.5244 | 3.1504 | 0.019402 |
| LaBraM/zuco1/real | 0.1689 | 2.1959 | 4.3919 | 0.022326 |
| LaBraM/zuco1/gaussian | 0.6757 | 2.3649 | 4.7297 | 0.028627 |
| LaBraM/zuco1/shuffled | 0.3378 | 2.0270 | 4.2230 | 0.021823 |
| LaBraM/zuco2/real | 0.4069 | 2.2380 | 3.9674 | 0.023559 |
| LaBraM/zuco2/gaussian | 0.5086 | 1.9329 | 3.2553 | 0.022214 |
| LaBraM/zuco2/shuffled | 0.1017 | 0.9156 | 1.8311 | 0.015474 |
| Unseen-content seed mean/zuco1/real | 2.3353 | 9.6564 | 17.2058 | 0.085268 |
| Unseen-content seed mean/zuco1/gaussian | 2.3353 | 9.0047 | 19.1742 | 0.086335 |
| Unseen-content seed mean/zuco1/shuffled | 2.8947 | 9.3204 | 16.5590 | 0.090252 |
| Unseen-content seed mean/zuco2/real | 1.3371 | 8.2486 | 16.5042 | 0.073448 |
| Unseen-content seed mean/zuco2/gaussian | 1.6911 | 8.5715 | 15.9886 | 0.075441 |
| Unseen-content seed mean/zuco2/shuffled | 1.3477 | 7.0727 | 13.1796 | 0.068180 |
| cnn_candidates/batch/real | 0.3378 | 1.5203 | 3.2095 | 0.020890 |
| cnn_candidates/batch/gaussian | 0.5068 | 2.0270 | 3.8851 | 0.022861 |
| cnn_candidates/batch/shuffled | 0.5068 | 1.3514 | 3.3784 | 0.022480 |
| cnn_candidates/training_unique/real | 0.3378 | 1.6892 | 3.0405 | 0.020789 |
| cnn_candidates/training_unique/gaussian | 0.3378 | 2.0270 | 3.8851 | 0.022962 |
| cnn_candidates/training_unique/shuffled | 0.3378 | 1.3514 | 3.2095 | 0.020752 |

The complete original export and separate CNN-loss outcomes are retained in [historical_export.csv](evidence_review_v1_20261101/historical_export.csv), with original JSON paths, seeds and pool sizes. The separately stored ZJS_validation JSON is a manifest/data audit without retrieval scores; it is [classified separately](evidence_review_v1_20261101/nonperformance_artifact.json) and excluded from performance tables. Duplicate timestamped LaBraM reports have equal saved values and are displayed once, not treated as replications. Statistical-ridge robustness and the adapted statistical comparator remain in the complete export; they are not silently discarded.

## Exploratory cross-release transfer: a fourth scope

Transfer evaluates the full target release, not an isolated held-out target test cohort. The following are equal-seed means of the saved wider-grid runs. Source-validation scores select ridge regularization. Cross-release participant independence is unresolved; inclusive pools retain shared source/target texts. The original-grid seed-42 runs and wider-grid runs are separate experiments and must not be pooled as repeats.

| Direction / condition (seed mean) | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| zuco1_to_zuco2/real | 0.2776 | 1.3879 | 2.8269 | 0.018529 |
| zuco1_to_zuco2/gaussian | 0.3456 | 1.4106 | 2.9175 | 0.019346 |
| zuco1_to_zuco2/shuffled | 0.2493 | 1.3540 | 2.7929 | 0.018247 |
| zuco2_to_zuco1/real | 0.5076 | 2.0106 | 3.8121 | 0.023592 |
| zuco2_to_zuco1/gaussian | 0.4578 | 1.9807 | 3.7723 | 0.023142 |
| zuco2_to_zuco1/shuffled | 0.3683 | 1.6821 | 3.2647 | 0.021216 |

Forward real-minus-Gaussian MRR is negative at every saved seed. Reverse differences change sign. Seed-level and participant-level differences are in [transfer_differences.json](evidence_review_v1_20261101/transfer_differences.json). Repeated seeds reuse target readers and texts, so they do not establish independent cohorts.

## Real-minus-control differences and reader consistency

Recall differences are percentage points; MRR differences remain reciprocal-rank units. Positive-reader counts describe paired differences on the same participants and are not significance tests. Full differences against every comparator and per-reader values are in [control_differences.json](evidence_review_v1_20261101/control_differences.json) and [participant_metrics.csv](evidence_review_v1_20261101/participant_metrics.csv).

| Scope / real arm minus comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Positive MRR readers |
|---|---:|---:|---:|---:|---:|---:|
| gru: real_eeg - matched_gaussian | +0.1382 | +0.0679 | -0.0617 | +0.001721 | 3/8 | 6/8 |
| gru: real_eeg - structural_only | -15.5705 | -38.5277 | -48.5809 | -0.257222 | 0/8 | 0/8 |
| gru_ce: real_eeg - matched_gaussian | +0.2688 | +0.0712 | -0.1037 | +0.002294 | 3/8 | 5/8 |
| gru_ce: real_eeg - structural_only | -15.5836 | -38.9893 | -49.3624 | -0.259490 | 0/8 | 0/8 |
| word_pretrain: pretrained_real - fresh_equal_steps | +0.0461 | -0.1977 | +0.0962 | +0.000294 | 3/8 | 4/8 |
| word_pretrain: pretrained_real - fresh_same_sentence | -0.1349 | -0.1719 | +0.0829 | -0.001169 | 0/8 | 1/8 |
| word_pretrain: pretrained_real - pretrained_gaussian | +0.1785 | +0.3584 | +0.6909 | +0.002702 | 4/8 | 5/8 |
| word_pretrain: pretrained_real - pretrained_missingness | -0.5031 | -1.4646 | -2.1001 | -0.010364 | 0/8 | 1/8 |
| word_pretrain: pretrained_real - structural_only | -15.7183 | -38.9050 | -49.0664 | -0.260122 | 0/8 | 0/8 |
| amplitude/semantic_embedding: structural_real - structural_only | +0.0000 | +0.0000 | +0.0000 | +0.000000 | 0/8 | 0/8 |
| amplitude/semantic_embedding: structural_real - structural_gaussian | +0.0000 | +0.0000 | +0.0000 | +0.000000 | 0/8 | 0/8 |
| amplitude/known_sentence_class: structural_real - structural_only | +0.0000 | +0.0000 | +0.0000 | +0.000000 | 0/8 | 0/8 |
| amplitude/known_sentence_class: structural_real - structural_gaussian | +0.0000 | +0.0000 | +0.0000 | +0.000000 | 0/8 | 0/8 |
| outer/incremental: structural_real - structural_only | -0.3906 | +0.3906 | +0.1953 | +0.000299 | 1/2 | 1/2 |
| outer/incremental: structural_real - structural_gaussian | -0.3906 | +0.1953 | +0.3906 | +0.000154 | 1/2 | 1/2 |
| outer/word_order: ordered_real - mean_real | +0.1953 | -0.3906 | +0.7812 | +0.002669 | 1/2 | 1/2 |
| outer/word_order: ordered_real - matched_gaussian | -0.3906 | -0.5859 | +0.0000 | -0.004469 | 0/2 | 0/2 |
| outer/word_order: ordered_real - structural_only | -17.1875 | -39.0625 | -47.2656 | -0.268917 | 0/2 | 0/2 |

The incremental-validation real-minus-structure MRR effects are zuco1:ZJS: -0.003530, zuco1:ZKW: +0.003092. Opposite signs prevent interpreting the small aggregate gain as consistent EEG information. Structural count/duration retrieval is a separate, strong annotation-derived finding. A model using real EEG without a structural block is not a direct incremental ablation; the combined TRT analyses supply that comparison.


### Historical test controls: participant consistency

These paired differences use the same saved test queries/candidate pools within each comparison. They do not supply new test evaluations or independent replications.

| Historical real minus control | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Positive MRR readers |
|---|---:|---:|---:|---:|---:|---:|
| historical/CNN/zuco1 - gaussian | +0.0000 | +0.0000 | -0.6757 | -0.001486 | 0/2 | 1/2 |
| historical/CNN/zuco1 - shuffled | +0.1689 | +0.1689 | -0.5068 | +0.000204 | 1/2 | 1/2 |
| historical/CNN/zuco2 - gaussian | +0.0000 | -0.7114 | -0.7114 | -0.002870 | 1/3 | 1/3 |
| historical/CNN/zuco2 - shuffled | +0.1016 | -0.1016 | +0.4065 | +0.000340 | 2/3 | 2/3 |
| historical/LaBraM/zuco1 - gaussian | -0.5068 | -0.1689 | -0.3378 | -0.006301 | 0/2 | 1/2 |
| historical/LaBraM/zuco1 - shuffled | -0.1689 | +0.1689 | +0.1689 | +0.000503 | 0/2 | 1/2 |
| historical/LaBraM/zuco2 - gaussian | -0.1017 | +0.3052 | +0.7121 | +0.001345 | 1/3 | 1/3 |
| historical/LaBraM/zuco2 - shuffled | +0.3052 | +1.3225 | +2.1363 | +0.008085 | 2/3 | 3/3 |
| historical/SSL/zuco1 - gaussian | -0.1689 | +0.0000 | -0.5068 | -0.002478 | 0/2 | 0/2 |
| historical/SSL/zuco1 - shuffled | -0.1689 | -0.1689 | +0.5068 | +0.000839 | 0/2 | 1/2 |
| historical/SSL/zuco2 - gaussian | +0.0000 | +0.1016 | +0.1016 | +0.000729 | 1/3 | 1/3 |
| historical/SSL/zuco2 - shuffled | +0.0000 | +0.0000 | +0.1016 | -0.000556 | 0/3 | 2/3 |
| historical/cnn_candidates/batch - gaussian | -0.1689 | -0.5068 | -0.6757 | -0.001971 | 0/2 | 0/2 |
| historical/cnn_candidates/batch - shuffled | -0.1689 | +0.1689 | -0.1689 | -0.001590 | 0/2 | 0/2 |
| historical/cnn_candidates/training_unique - gaussian | +0.0000 | -0.3378 | -0.8446 | -0.002173 | 1/2 | 1/2 |
| historical/cnn_candidates/training_unique - shuffled | +0.0000 | +0.3378 | -0.1689 | +0.000037 | 0/2 | 2/2 |

## Word task: separate supervised outcome

Exact exported labels and saved word TRT rows were paired across the declared training cohort before fitting. Text supplies target classes only. Each fold builds its vocabulary from seven fit readers, retains missing slots, and scores any out-of-vocabulary held-out target as a miss with reciprocal rank zero. No difficult sentence is excluded. These reader splits repeat sentence classes; no unseen-word generalization is established.

| Word-stage condition (macro across readers) | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| pretrained_class_shuffled | 0.0668 | 0.1988 | 0.3240 | 0.003046 |
| pretrained_gaussian | 3.0019 | 9.7271 | 15.5035 | 0.067224 |
| pretrained_missingness | 3.3331 | 14.5043 | 21.8359 | 0.088084 |
| pretrained_real | 3.0075 | 10.0459 | 15.9434 | 0.068325 |

These are word-class scores, not sentence accuracy. The missingness-only word model is stronger than real EEG here. Frequent lexical classes and gaze-derived missingness can support these scores without demonstrated EEG language information. Per-reader word outcomes are in [word_task.csv](evidence_review_v1_20261101/word_task.csv).


Word real-minus-control differences are also saved separately in [word_control_differences.json](evidence_review_v1_20261101/word_control_differences.json).

| Word-stage real minus comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers |
|---|---:|---:|---:|---:|---:|
| pretrained_class_shuffled | +2.9408 | +9.8471 | +15.6194 | +0.065280 | 8/8 |
| pretrained_gaussian | +0.0056 | +0.3188 | +0.4400 | +0.001101 | 4/8 |
| pretrained_missingness | -0.3256 | -4.4584 | -5.8924 | -0.019759 | 1/8 |

## Expected N-way retrieval: training-reader CV expectations

This analysis postdates the 2026-11-01 evidence review. It replays the frozen word-pretrained models on the same eight training-reader folds and adds no training, tuning, outer-validation or test evaluation. Each pool holds the correct sentence plus N−1 uniformly drawn distractors from the original 299. Values are exact expected top-1 percentages, averaged within reader and then equally across readers; ties with the correct sentence count as misses. They are not a new cohort, a Monte Carlo estimate or a model improvement.

| N | Pretrained real EEG % | Gaussian % | Class-shuffled % | Missingness-only % | Structure-only % | Chance % |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 49.2079 | 49.2034 | 50.2292 | 53.8270 | 90.4389 | 50.0000 |
| 5 | 19.5309 | 18.8058 | 20.2595 | 23.7697 | 75.0557 | 20.0000 |
| 10 | 9.7124 | 8.8002 | 10.2744 | 12.8284 | 62.8101 | 10.0000 |
| 20 | 4.7439 | 4.0531 | 5.2801 | 7.0310 | 51.3648 | 5.0000 |
| 50 | 1.8078 | 1.4160 | 2.2851 | 3.2405 | 37.8816 | 2.0000 |
| 100 | 0.8882 | 0.6483 | 1.2739 | 1.8675 | 28.5345 | 1.0000 |
| 300 | 0.3946 | 0.2160 | 0.5783 | 0.8977 | 16.1129 | 0.3333 |

Smaller N raises chance and changes difficulty. Compare each column with chance and with controls at the same N: two-way real EEG is below 50% chance, and missingness exceeds real EEG at every N. Random distractors do not control sentence length or semantic similarity, so structural scores remain annotation-derived. [N-way archive](nway_word_pretrain_v1_20261101/README.md); [summary.csv](nway_word_pretrain_v1_20261101/summary.csv).

## Signal quality and robustness

The only corrupted-input evaluations in the [original export](retrieval.csv) are ZuCo2 ridge runs on one split and one corruption seed, with 984 queries and 344 candidates. Clean MRR is 0.023036 for standard features and 0.020230 with the quality heuristic. Quality weighting is lower in 8 of the 9 nonzero conditions; 25% channel dropout is the exception. Scores stay near chance, so these nonmonotonic curves do not establish a robustness effect. CNN, SSL, LaBraM and later pilots have no degraded-input evaluation. [Figure](quality_robustness_mrr.png).

| Corruption | Severity | Standard MRR | Quality-aware MRR | Difference |
|---|---:|---:|---:|---:|
| additive noise | 0.25 | 0.022548 | 0.020930 | -0.001618 |
| additive noise | 0.5 | 0.022272 | 0.017658 | -0.004615 |
| additive noise | 1.0 | 0.021292 | 0.019364 | -0.001928 |
| channel dropout | 0.1 | 0.021798 | 0.016228 | -0.005570 |
| channel dropout | 0.25 | 0.020604 | 0.022361 | +0.001757 |
| channel dropout | 0.5 | 0.021021 | 0.016720 | -0.004301 |
| temporal mask | 0.1 | 0.023398 | 0.019309 | -0.004089 |
| temporal mask | 0.25 | 0.023623 | 0.020671 | -0.002952 |
| temporal mask | 0.5 | 0.024529 | 0.021663 | -0.002866 |

## Selection history and limitations

Spectral arms searched 28 settings each on outer-validation MRR; incremental combined arms searched 196 each, versus 28 for structure. Ordered-word ridge selected among three alphas using eight training-reader folds. The amplitude audit selected among three alphas and three EEG weights for combined arms (three alphas for individual blocks), on the same readers later summarized. Both GRU objectives selected a shared epoch from 4/8/12 by real-EEG training-CV Recall@1, preferring earlier ties; every control used that epoch. Both chose epoch 8. This gives real EEG selection opportunity and makes the reported CV scores optimistic development descriptions.

Word pretraining instead fixed 128 word updates plus 256 sentence updates with no epoch search. The same-sentence fresh baseline has 256 sentence updates; the equal-update baseline has 384 sentence updates with an optimizer reset after 128. Equal update counts do not match FLOPs or sequence exposure. Only the projection transfers; the word head and optimizer state are discarded and GRU/output-head initialization is shared.

All later downstream scalers, Gaussian moments and vocabularies fit fold-training readers only; masks and missing-word positions remain separate from padding. However, earlier experiments already informed hypotheses and representation choices. Overlapping folds, repeated words/sentences, reused validation and historical test inspection are not independent cohorts. No aggregate final accuracy, causal explanation, independent confirmation or word-level confidence interval follows.

The measurement remains gaze/stimulus assisted. Source-session filtering/reference histories, exact producer revisions, gamma2 definition discrepancies and possible physical boundary tails remain unresolved. Historical waveform Gaussian controls match each observation/channel standard deviation, retaining observation-specific amplitude information. Later TRT Gaussian controls use moments fitted on training words and preserve exact word counts and missingness. These are different nulls; neither preserves full real EEG covariance/distribution. Cross-release participant identities remain unverified; strict independent transfer remains unestablished. Local execution does not establish an independent cloud reproduction.

## Verified findings versus explanations

**Verified engineering/evidence findings:** exported text can differ from a stimulus CSV without mismatching the selected source/cache pair; the proposed punctuation repair remains unapplied. Source word/cache pairing and saved-rank reproduction passed the earlier recorded audits. Interpreter/sandbox access and a historical large-fixture storage failure were execution constraints, not evidence of bad EEG labels. The earlier report retains provenance for superseded ridge artifacts, which must not be counted as independent successful methods.

**Observed diagnostics, not confirmed failure causes:** reader amplitude norms differ; a relative transform reduces that spread; PCA retains only part of standardized variance; positive-only GRU predictions concentrate on a few candidates; full-candidate loss spreads predictions. None proves that amplitude, PCA loss, coarse sentence supervision or objective concentration caused poor retrieval. The bounded follow-ups did not demonstrate the desired improvement. No confirmed current pairing/ranking bug was identified in these reviewed paths. Low scores do not prove incorrect pairing or absence of language information in EEG.

## Project brief coverage

"Done" means the step was executed and reported, not that decoding succeeded.

| Brief step | Evidence | Status |
|---|---|---|
| 1. Data collection and preparation | ZuCo 1.0 and 2.0 prepared; duration-QC cohorts of 3,349 and 5,884 observations; reader-disjoint train, validation and test splits | Done |
| 2. EEG preprocessing | 105 ordered channels, 200 Hz, training-fit normalizers; upstream session processing is inherited and cannot be refit | Done |
| 3. EEG representation learning | Frozen adapted LaBraM, reconstruction SSL trained on ZuCo, supervised CNN and statistical baselines; LaBraM used an approximate 19-electrode mapping without fine-tuning; no useful advantage | Done |
| 4. EEG-to-language decoding | Fixed-pool retrieval against sentence embeddings with Gaussian and shuffled controls; controlled negative result | Done |
| 5. Signal quality and robustness | Three corruption types and a quality heuristic, on ZuCo2 ridge only with one split and corruption seed | Partial |
| 6. New subjects and datasets | Unseen readers tested in both releases; cross-release transfer is exploratory only; strict participant-independent transfer was not run | Partial |
| 7. Evaluation and analysis | Methods compared with controls on clean inputs and limitations documented; degraded-condition analysis covers ridge only | Partial |

## Block-holdout pilot v2: 2026-11-05

A later frozen pilot kept neighbouring sentences out of training. No model passed the gate on validation, so its test blocks were not scored. At 10-way, the CNN on real EEG scored 10.02% against 9.72% on matched noise, ridge F scored 12.66% against 12.41%, and word counts and timing with no EEG scored 13.03%. The earlier random-holdout pilot scored 23.10% on validation, but that difference also reflects fewer training sentences and different preprocessing, so it does not measure the leak alone. [Full v2 results](block_holdout_pilot_v2_20261105/REPORT.md).

## Review decision and reproducibility

Stop experiments under the current instruction. Further work would need a revised, falsifiable hypothesis and a narrower question with controls, or better independent reader/stimulus data for a generalization claim. Reusing these development scores for another broad search would not supply independent evidence. There is no authorization here for more fitting, target repair, feature extraction, evaluation or correspondence.

[Supervisor-review note](evidence_review_v1_20261101/SUPERVISOR_NOTE.md). [Numerical/artifact checks](evidence_review_v1_20261101/verification.json). [Source JSON hashes](evidence_review_v1_20261101/source_hashes.json). [Byte-preservation check](evidence_review_v1_20261101/preservation_after.json). The earlier report and presentation are archived, and all previous experiment files remain intact. Historical software-test counts in the archived report are historical; this consolidation runs artifact and numerical checks only.
