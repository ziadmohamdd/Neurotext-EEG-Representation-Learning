# GRU full-candidate retrieval-loss comparison — training CV only

This is one fixed loss-only comparison against the preserved cosine GRU baseline. All scores are selected equal-reader training-CV results on eight original training readers, 2,245 observations and 300 candidates; no outer validation or test evaluation occurred.

## Diagnostic motivation

Replaying all 48 selected baseline checkpoints reproduced every saved rank. With real EEG, the cosine GRU chose 6.375 distinct top-1 sentences per reader on average, and its largest top-1 share averaged 62.96%. Mean cosine to the original training target centroid was 0.923457. This is observed prediction concentration, not a proven cause or a pairing bug. Weak input information can also produce concentration; spreading predictions does not by itself demonstrate useful EEG information.

## Fixed comparison

The 45,952-parameter 841→32 projection, 32-unit GRU and 32→384 output head are unchanged. All original model initial tensors and preprocessing arrays match the cosine run exactly. Data order, masks, word counts, source word ordering, target embeddings, noise draws, optimizer (AdamW, LR .001, decay .01), batch size 64, clipping 1, seed 42, 12-epoch budget and checkpoints 4/8/12 remain fixed. No additional hyperparameter search occurred.

Only the neural loss changes: instead of positive-only mean(1−cosine), normalized predicted embeddings score every unique fold-training candidate, divided by fixed temperature .07, and cross-entropy supervises the correct class index. Repeated target observations do not duplicate candidates or create false negatives. Each fold has all 300 training classes; this is checked before fitting. Word text/identity never enters the EEG feature matrix.

The common epoch is selected by real-EEG macro training-CV Recall@1 from 4/8/12, with earlier epochs winning ties; every neural control uses that epoch. This procedure gives real EEG selection opportunity and produces optimistic development scores. Controls are not independently epoch-optimized. Both objective runs use the same rule. The report also compares the old objective at the new objective's selected epoch to avoid presenting an epoch difference as a loss effect. No outer arrays are read regardless of the gate.

Cosine selected epoch: **8**. Full-candidate CE selected epoch: **8**.

## Full-candidate CE selected training-CV results

Recall is percent; MRR is a reciprocal-rank score, not accuracy.

| Condition | Macro R@1 % | Macro R@5 % | Macro R@10 % | Macro MRR |
|---|---:|---:|---:|---:|
| real_eeg | 0.5293 | 1.2973 | 2.6902 | 0.020781 |
| mean_eeg | 0.3630 | 1.4867 | 2.9624 | 0.019303 |
| matched_gaussian | 0.2605 | 1.2261 | 2.7938 | 0.018486 |
| class_shuffled | 0.1744 | 1.5069 | 3.7394 | 0.020120 |
| word_order_shuffled | 0.2650 | 1.3330 | 2.6156 | 0.018809 |
| missingness_only | 0.4490 | 1.9321 | 3.9432 | 0.023184 |
| structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| Uniform ranking expectation | 0.3333 | 1.6667 | 3.3333 | 0.020942 |

## Objective comparison for real EEG

| Evaluation point | Macro R@1 % | Macro MRR |
|---|---:|---:|
| Cosine at its selected epoch 8 | 0.5424 | 0.023048 |
| Cosine at matched epoch 8 | 0.5424 | 0.023048 |
| Full-candidate CE at selected epoch 8 | 0.5293 | 0.020781 |

## Real minus comparators

| Comparator | ΔR@1 pp | ΔMRR | Readers with positive R@1 gap |
|---|---:|---:|---:|
| mean_eeg | +0.1663 | +0.001477 | 3/8 |
| matched_gaussian | +0.2688 | +0.002294 | 3/8 |
| class_shuffled | +0.3549 | +0.000660 | 4/8 |
| word_order_shuffled | +0.2643 | +0.001971 | 4/8 |
| missingness_only | +0.0804 | -0.002403 | 2/8 |
| structural_only | -15.5836 | -0.259490 | 0/8 |

## Prediction concentration and decision

New real-EEG outputs choose 53.000 distinct top-1 sentences per reader on average; the largest top-1 share averages 17.82%. Interpret this together with control gaps, not as success by itself.

Prespecified descriptive gate passed: **False**. The inherited gate requires at least +2 pp R@1, +.005 MRR, and positive R@1 differences in six of eight readers against every comparator, including structure. It is not a significance test. Stop after this bounded training-only run; no search expansion or outer/test evaluation.

## Verification and limitations

Twenty focused tests passed before fitting. Final checks replayed all 48 selected models and reproduced 13,470 ranks; initialization and all preprocessing arrays match baseline, and the structural comparator predictions match exactly. 2003 prior files remain SHA256-identical; raw dataset size/time inventory is unchanged. Existing code changes, datasets, targets, checkpoints, results, main report, presentation and .venv are preserved.

One loss, temperature and seed were tested. Overlapping training folds and repeated sentences are not independent cohorts. These readers have extensive development history. Source fixation/stimulus-assisted alignment, structural shortcuts, upstream preprocessing and provenance limits remain. The Gaussian null preserves masks/counts and raw marginal moments but not full real-data covariance/distribution. This comparison does not establish unseen-sentence generalization or a validated brain-to-text decoder.

Negative-candidate training is a general representation-learning idea; [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) studies contrastive text embeddings. This EEG experiment is not a reproduction of its model or performance.

## Per-reader selected CE results

| Condition | Training holdout reader | R@1 % | R@5 % | R@10 % | MRR |
|---|---|---:|---:|---:|---:|
| real_eeg | ZDM | 0.3460 | 1.0381 | 3.1142 | 0.021107 |
| real_eeg | ZDN | 1.3468 | 2.0202 | 2.6936 | 0.026758 |
| real_eeg | ZGW | 1.4235 | 2.4911 | 4.6263 | 0.034660 |
| real_eeg | ZJM | 0.7782 | 2.7237 | 5.8366 | 0.031267 |
| real_eeg | ZJN | 0.0000 | 0.3401 | 1.0204 | 0.010983 |
| real_eeg | ZKH | 0.3401 | 1.3605 | 2.7211 | 0.020381 |
| real_eeg | ZMG | 0.0000 | 0.0000 | 0.6993 | 0.010417 |
| real_eeg | ZPH | 0.0000 | 0.4049 | 0.8097 | 0.010670 |
| mean_eeg | ZDM | 0.6920 | 1.7301 | 2.7682 | 0.025097 |
| mean_eeg | ZDN | 0.6734 | 1.6835 | 3.0303 | 0.022821 |
| mean_eeg | ZGW | 0.3559 | 2.8470 | 5.6940 | 0.024805 |
| mean_eeg | ZJM | 0.7782 | 3.1128 | 5.4475 | 0.029468 |
| mean_eeg | ZJN | 0.0000 | 0.6803 | 1.0204 | 0.010974 |
| mean_eeg | ZKH | 0.0000 | 0.6803 | 2.7211 | 0.015720 |
| mean_eeg | ZMG | 0.0000 | 0.3497 | 1.3986 | 0.010212 |
| mean_eeg | ZPH | 0.4049 | 0.8097 | 1.6194 | 0.015329 |
| matched_gaussian | ZDM | 0.3460 | 1.0381 | 2.0761 | 0.017019 |
| matched_gaussian | ZDN | 0.3367 | 1.0101 | 2.3569 | 0.020714 |
| matched_gaussian | ZGW | 0.7117 | 2.8470 | 3.9146 | 0.026794 |
| matched_gaussian | ZJM | 0.0000 | 0.3891 | 1.9455 | 0.013499 |
| matched_gaussian | ZJN | 0.0000 | 1.3605 | 2.3810 | 0.016632 |
| matched_gaussian | ZKH | 0.3401 | 1.3605 | 3.4014 | 0.019242 |
| matched_gaussian | ZMG | 0.3497 | 1.3986 | 3.8462 | 0.020272 |
| matched_gaussian | ZPH | 0.0000 | 0.4049 | 2.4291 | 0.013720 |
| class_shuffled | ZDM | 0.0000 | 1.0381 | 1.7301 | 0.016150 |
| class_shuffled | ZDN | 0.0000 | 1.3468 | 3.3670 | 0.019541 |
| class_shuffled | ZGW | 0.3559 | 1.7794 | 4.2705 | 0.023545 |
| class_shuffled | ZJM | 0.0000 | 1.1673 | 3.8911 | 0.019272 |
| class_shuffled | ZJN | 0.0000 | 1.0204 | 3.4014 | 0.017178 |
| class_shuffled | ZKH | 0.3401 | 2.0408 | 4.4218 | 0.021162 |
| class_shuffled | ZMG | 0.6993 | 2.4476 | 5.5944 | 0.025805 |
| class_shuffled | ZPH | 0.0000 | 1.2146 | 3.2389 | 0.018309 |
| word_order_shuffled | ZDM | 0.3460 | 2.0761 | 2.7682 | 0.024312 |
| word_order_shuffled | ZDN | 0.6734 | 1.6835 | 3.7037 | 0.022377 |
| word_order_shuffled | ZGW | 0.7117 | 2.1352 | 4.9822 | 0.028634 |
| word_order_shuffled | ZJM | 0.3891 | 1.9455 | 3.5019 | 0.021935 |
| word_order_shuffled | ZJN | 0.0000 | 0.0000 | 0.6803 | 0.008949 |
| word_order_shuffled | ZKH | 0.0000 | 1.0204 | 2.3810 | 0.015377 |
| word_order_shuffled | ZMG | 0.0000 | 1.3986 | 2.0979 | 0.015832 |
| word_order_shuffled | ZPH | 0.0000 | 0.4049 | 0.8097 | 0.013059 |
| missingness_only | ZDM | 0.3460 | 1.7301 | 4.1522 | 0.022202 |
| missingness_only | ZDN | 0.3367 | 1.0101 | 3.3670 | 0.019937 |
| missingness_only | ZGW | 0.3559 | 1.4235 | 3.9146 | 0.022205 |
| missingness_only | ZJM | 0.7782 | 2.3346 | 4.6693 | 0.027621 |
| missingness_only | ZJN | 0.3401 | 2.0408 | 3.4014 | 0.021288 |
| missingness_only | ZKH | 0.6803 | 2.0408 | 3.7415 | 0.024684 |
| missingness_only | ZMG | 0.3497 | 2.4476 | 3.8462 | 0.023505 |
| missingness_only | ZPH | 0.4049 | 2.4291 | 4.4534 | 0.024028 |
| structural_only | ZDM | 14.8789 | 40.1384 | 51.9031 | 0.270690 |
| structural_only | ZDN | 12.7946 | 39.7306 | 52.1886 | 0.254211 |
| structural_only | ZGW | 18.1495 | 39.5018 | 51.6014 | 0.290449 |
| structural_only | ZJM | 15.5642 | 39.6887 | 51.3619 | 0.275772 |
| structural_only | ZJN | 17.3469 | 37.0748 | 50.0000 | 0.277953 |
| structural_only | ZKH | 16.3265 | 40.4762 | 52.0408 | 0.285932 |
| structural_only | ZMG | 16.4336 | 41.9580 | 51.0490 | 0.285010 |
| structural_only | ZPH | 17.4089 | 43.7247 | 56.2753 | 0.302149 |

Full histories, scored epochs, candidate pools, fold identities, controls, configurations and source snapshots are archived here. Coefficients/checkpoints remain in the new run folder.
