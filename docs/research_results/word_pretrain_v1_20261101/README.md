# Exported-word supervision before sentence retrieval — v1

**Training-reader CV only. Feasibility passed. Prespecified descriptive gate: False. Fixed run ended; no outer-validation or test evaluation.**

This bounded word-pretraining pilot did not establish a useful, consistent sentence-retrieval improvement beyond the fresh baselines and controls.

## Source feasibility and unchanged targets

All 2,245 original training observations and 48,865 ordered word slots were checked directly against the eight training-reader source MAT files. For every selected slot, joining exported word.content with spaces exactly reproduced the unchanged parent sentence/manifest target. All eight-band × 105-channel TRT values and every missing position matched the saved word cache exactly. Source hashes match the prior audited sources. The exported word order is stimulus order, not chronological fixation order. No punctuation repair, lowercasing, regex filtering, token splitting or stimulus CSV substitution was applied.

The documented mapping is in docs/word_eeg_feasibility_20261030.md, especially “Sentence-word and fixation mapping evidence.” The new opt-in bounded reader materializes only requested training slots and only parent content, exported word content and the eight TRT vectors. It skips raw waveforms, raw ET, gaze coordinates, durations, word bounds, fixations and all other fields. No outer-validation/test source file was opened. Trustworthy exported row/label pairing does not independently certify gaze assignment, physiological quality or physical sentence containment.

## Sentence retrieval: fixed training-CV results

Equal-reader macro scores over the original eight reader-disjoint holdout folds, 2,245 observations and unchanged 300-candidate pool. Recall is percent. MRR is not accuracy. Every sentence observation is retained, including all-missing ZMG/70. All scores use the final fixed checkpoint; no epoch or setting was selected from these outcomes.

| Arm | Word updates | Sentence updates | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|---:|---:|
| pretrained_real | 128 | 256 | 0.3946 | 1.3816 | 2.9861 | 0.020149 |
| fresh_equal_steps | 0 | 384 | 0.3485 | 1.5793 | 2.8899 | 0.019855 |
| fresh_same_sentence | 0 | 256 | 0.5294 | 1.5535 | 2.9032 | 0.021318 |
| pretrained_gaussian | 128 | 256 | 0.2160 | 1.0232 | 2.2952 | 0.017447 |
| pretrained_class_shuffled | 128 | 256 | 0.5783 | 1.8241 | 3.7205 | 0.023627 |
| pretrained_missingness | 128 | 256 | 0.8977 | 2.8463 | 5.0861 | 0.030513 |
| structural_only | 0 | 0 | 16.1129 | 40.2867 | 52.0525 | 0.280271 |

## Pretrained real EEG minus comparators

| Comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Gate |
|---|---:|---:|---:|---:|---:|---|
| fresh_equal_steps | +0.0461 | -0.1977 | +0.0962 | +0.000294 | 3/8 | False |
| fresh_same_sentence | -0.1349 | -0.1719 | +0.0829 | -0.001169 | 0/8 | False |
| pretrained_gaussian | +0.1785 | +0.3584 | +0.6909 | +0.002702 | 4/8 | False |
| pretrained_class_shuffled | -0.1838 | -0.4424 | -0.7344 | -0.003478 | 3/8 | False |
| pretrained_missingness | -0.5031 | -1.4646 | -2.1001 | -0.010364 | 0/8 | False |
| structural_only | -15.7183 | -38.9050 | -49.0664 | -0.260122 | 0/8 | False |

Before fitting, the gate required pretrained real EEG to exceed every comparator by at least 2 percentage points Recall@1 and 0.005 MRR, with positive Recall@1 differences for at least six of eight readers against every comparator. This is a descriptive screen, not a significance test. The protocol stops after training CV regardless of the gate.

## Word task: separate from sentence retrieval

Word text is supervised target metadata only. Each fold builds an exact-string vocabulary from its seven fit readers only. The targets are fixed one-hot classes; no text embedding model, external vocabulary or target-side learned transform is used. Case and punctuation remain part of the class. Held-out labels never expand the vocabulary. Every held-out slot is retained; out-of-vocabulary targets receive a miss and reciprocal rank zero. Missing EEG slots also remain in both word training and evaluation, with zero imputed EEG plus an observed-word indicator. Word accuracy must not be read as sentence accuracy.

| Arm | Macro word R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| pretrained_real | 3.0075 | 10.0459 | 15.9434 | 0.068325 |
| pretrained_gaussian | 3.0019 | 9.7271 | 15.5035 | 0.067224 |
| pretrained_class_shuffled | 0.0668 | 0.1988 | 0.3240 | 0.003046 |
| pretrained_missingness | 3.3331 | 14.5043 | 21.8359 | 0.088084 |

| Holdout reader | Fold-training word classes | Held-out word slots | OOV slots | Missing EEG slots |
|---|---:|---:|---:|---:|
| ZDM | 2716 | 6269 | 0 | 2567 |
| ZDN | 2716 | 6525 | 0 | 3338 |
| ZGW | 2716 | 6027 | 0 | 2059 |
| ZJM | 2716 | 5434 | 0 | 1111 |
| ZJN | 2716 | 6424 | 0 | 1084 |
| ZKH | 2716 | 6465 | 0 | 2237 |
| ZMG | 2716 | 6259 | 0 | 2247 |
| ZPH | 2716 | 5462 | 0 | 1374 |

Word metrics score all slots. Per-reader word results, observed-only and observed/in-vocabulary metrics, per-slot ranks, true target indices and missingness masks are saved separately in word_holdout.json and word_task_per_reader.csv. These reader folds repeat sentence classes; vocabulary coverage here must not be interpreted as evidence of unseen-word generalization. No word-level statistical independence or confidence interval is claimed. Frequent-word prediction can score above uniform chance without useful EEG information; matched controls are required.

## Architecture, transfer and training budgets

The unchanged sentence architecture has 45,952 parameters: 841→32 linear projection (26,944), tanh, one 32-unit GRU (6,336), and a 32→384 sentence-embedding head (12,672). The word encoder uses the same 841→32 projection and tanh with a new 32→V linear classification head. Its parameter count is 26,944 + 33V, where V is the fold-training vocabulary size, recorded for each fold. The only transferred tensors are project.weight (32×841) and project.bias (32), totaling 26,944 parameters. The word head is discarded. The GRU, sentence head and optimizer state do not transfer; GRU and sentence head retain the common original initialization. Every transfer was checked tensor by tensor.

Numeric inputs contain only 840 source EEG coordinates and one observed-word indicator. No text, word IDs, sentence IDs, subject IDs, target vectors or eye-tracking measurements enter model input tensors. The sentence loss uses unchanged cached sentence embeddings only on its target/candidate side. EEG undergoes the existing signed log1p, per-word/per-band 105-channel centering and fold-training complete-word standardization. Missing slots become scaled training-mean zeros and remain recurrent steps. True word lengths, rather than valid-word counts, drive packed sequences; padded tails are excluded. All-missing observations retain their original length and learnable bias/length fallback.

Seed 42, projection/hidden width 32, AdamW LR .001, weight decay .01, gradient clipping 1, word minibatches 256 and sentence minibatches up to 64 are fixed. The word stage runs exactly 128 optimizer updates with cross entropy over fold-training word classes. The sentence stage runs exactly 256 updates with full-candidate cosine-logit cross entropy at temperature .07. Batch orders use deterministic repeated full permutations with short end-of-cycle batches. Every update loss and batch size is saved. AdamW resets at the stage boundary. Only final stage checkpoints are scored; no early stopping or hyperparameter search occurred.

The same-sentence fresh baseline trains a newly initialized GRU for 256 sentence updates without word training. The equal-step fresh baseline trains for 128 sentence updates, resets AdamW, then runs the same 256-update sentence stage, totaling 384 updates. All pretrained arms use 128 word + 256 sentence updates, also totaling 384. Matching by optimizer-update count does not match FLOPs, sequence exposure or number of supervised examples: a word update uses independent 841-coordinate slots, while a sentence update processes complete sequences and updates the GRU. Both baselines are reported to separate these budget effects; the historical selected-epoch GRU scores are context, not the fresh baseline.

Gaussian, class-shuffled and missingness-only arms use exactly the same two stages, architecture, initial tensors, update budgets and optimizers. Gaussian draws preserve exact word counts and coordinate missingness; raw marginal moments fit fold-training words only, and Gaussian scaling fits Gaussian fold-training words. The Gaussian null does not preserve real covariance/full distribution. Class shuffling consistently deranges exact word classes during word training and sentence classes during sentence training, with correct held-out labels at both tasks; mappings are saved. This is a whole-supervision control. Missingness-only zeros all EEG amplitudes while preserving indicators and sentence lengths. Structural-only uses the same observations, first 14 count/duration summaries, fold-training count basis, training-fitted median/scaler and fixed ridge alpha 1. Its ranks exactly match the prior structural recipe on these folds.

## Per-reader sentence retrieval

| Arm | Reader | Queries | R@1 % | R@5 % | R@10 % | MRR |
|---|---|---:|---:|---:|---:|---:|
| pretrained_real | ZDM | 289 | 0.3460 | 1.3841 | 2.4221 | 0.019614 |
| pretrained_real | ZDN | 297 | 0.6734 | 1.0101 | 2.6936 | 0.021545 |
| pretrained_real | ZGW | 281 | 1.0676 | 2.1352 | 6.0498 | 0.032168 |
| pretrained_real | ZJM | 257 | 0.3891 | 3.1128 | 5.8366 | 0.028180 |
| pretrained_real | ZJN | 294 | 0.0000 | 0.6803 | 1.7007 | 0.012246 |
| pretrained_real | ZKH | 294 | 0.6803 | 2.3810 | 4.0816 | 0.026181 |
| pretrained_real | ZMG | 286 | 0.0000 | 0.3497 | 0.6993 | 0.011182 |
| pretrained_real | ZPH | 247 | 0.0000 | 0.0000 | 0.4049 | 0.010075 |
| fresh_equal_steps | ZDM | 289 | 0.0000 | 1.0381 | 2.7682 | 0.016171 |
| fresh_equal_steps | ZDN | 297 | 1.3468 | 1.6835 | 2.6936 | 0.026200 |
| fresh_equal_steps | ZGW | 281 | 0.7117 | 3.2028 | 4.9822 | 0.030332 |
| fresh_equal_steps | ZJM | 257 | 0.3891 | 4.6693 | 6.6148 | 0.031007 |
| fresh_equal_steps | ZJN | 294 | 0.0000 | 0.6803 | 1.0204 | 0.013206 |
| fresh_equal_steps | ZKH | 294 | 0.3401 | 1.3605 | 2.7211 | 0.020672 |
| fresh_equal_steps | ZMG | 286 | 0.0000 | 0.0000 | 0.6993 | 0.010276 |
| fresh_equal_steps | ZPH | 247 | 0.0000 | 0.0000 | 1.6194 | 0.010975 |
| fresh_same_sentence | ZDM | 289 | 0.3460 | 1.3841 | 2.4221 | 0.019912 |
| fresh_same_sentence | ZDN | 297 | 0.6734 | 2.6936 | 3.3670 | 0.024830 |
| fresh_same_sentence | ZGW | 281 | 1.0676 | 2.4911 | 5.3381 | 0.032778 |
| fresh_same_sentence | ZJM | 257 | 0.7782 | 2.7237 | 5.8366 | 0.031267 |
| fresh_same_sentence | ZJN | 294 | 0.0000 | 0.6803 | 1.3605 | 0.012250 |
| fresh_same_sentence | ZKH | 294 | 1.0204 | 1.7007 | 3.7415 | 0.025928 |
| fresh_same_sentence | ZMG | 286 | 0.3497 | 0.3497 | 0.3497 | 0.012906 |
| fresh_same_sentence | ZPH | 247 | 0.0000 | 0.4049 | 0.8097 | 0.010670 |
| pretrained_gaussian | ZDM | 289 | 0.3460 | 0.6920 | 3.1142 | 0.018171 |
| pretrained_gaussian | ZDN | 297 | 0.3367 | 1.3468 | 2.3569 | 0.018602 |
| pretrained_gaussian | ZGW | 281 | 0.3559 | 1.4235 | 2.4911 | 0.021246 |
| pretrained_gaussian | ZJM | 257 | 0.0000 | 0.7782 | 2.3346 | 0.014963 |
| pretrained_gaussian | ZJN | 294 | 0.0000 | 0.6803 | 1.7007 | 0.014496 |
| pretrained_gaussian | ZKH | 294 | 0.3401 | 1.7007 | 3.4014 | 0.018697 |
| pretrained_gaussian | ZMG | 286 | 0.3497 | 0.3497 | 1.7483 | 0.017341 |
| pretrained_gaussian | ZPH | 247 | 0.0000 | 1.2146 | 1.2146 | 0.016059 |
| pretrained_class_shuffled | ZDM | 289 | 0.3460 | 0.6920 | 2.4221 | 0.017720 |
| pretrained_class_shuffled | ZDN | 297 | 0.0000 | 1.6835 | 5.3872 | 0.021258 |
| pretrained_class_shuffled | ZGW | 281 | 0.7117 | 1.4235 | 3.9146 | 0.025931 |
| pretrained_class_shuffled | ZJM | 257 | 0.0000 | 1.9455 | 3.1128 | 0.019854 |
| pretrained_class_shuffled | ZJN | 294 | 0.0000 | 1.0204 | 2.7211 | 0.016395 |
| pretrained_class_shuffled | ZKH | 294 | 1.3605 | 3.0612 | 4.4218 | 0.031488 |
| pretrained_class_shuffled | ZMG | 286 | 1.3986 | 3.1469 | 4.5455 | 0.031157 |
| pretrained_class_shuffled | ZPH | 247 | 0.8097 | 1.6194 | 3.2389 | 0.025212 |
| pretrained_missingness | ZDM | 289 | 1.0381 | 3.1142 | 6.2284 | 0.031194 |
| pretrained_missingness | ZDN | 297 | 0.6734 | 2.3569 | 4.0404 | 0.027321 |
| pretrained_missingness | ZGW | 281 | 1.0676 | 2.4911 | 4.6263 | 0.030683 |
| pretrained_missingness | ZJM | 257 | 0.7782 | 2.7237 | 5.8366 | 0.031699 |
| pretrained_missingness | ZJN | 294 | 0.6803 | 4.0816 | 5.4422 | 0.032235 |
| pretrained_missingness | ZKH | 294 | 0.6803 | 2.7211 | 4.7619 | 0.029508 |
| pretrained_missingness | ZMG | 286 | 1.0490 | 2.4476 | 4.8951 | 0.028645 |
| pretrained_missingness | ZPH | 247 | 1.2146 | 2.8340 | 4.8583 | 0.032814 |
| structural_only | ZDM | 289 | 14.8789 | 40.1384 | 51.9031 | 0.270690 |
| structural_only | ZDN | 297 | 12.7946 | 39.7306 | 52.1886 | 0.254211 |
| structural_only | ZGW | 281 | 18.1495 | 39.5018 | 51.6014 | 0.290449 |
| structural_only | ZJM | 257 | 15.5642 | 39.6887 | 51.3619 | 0.275772 |
| structural_only | ZJN | 294 | 17.3469 | 37.0748 | 50.0000 | 0.277953 |
| structural_only | ZKH | 294 | 16.3265 | 40.4762 | 52.0408 | 0.285932 |
| structural_only | ZMG | 286 | 16.4336 | 41.9580 | 51.0490 | 0.285010 |
| structural_only | ZPH | 247 | 17.4089 | 43.7247 | 56.2753 | 0.302149 |

## Verification, preservation and limitations

Fourteen focused tests passed before fitting, covering source pairing, unchanged strings, training-only vocabulary/OOV/scaling, Gaussian masks, padding invariance, projection-only transfer, fixed budgets, loss and reader separation. Post-run replay verified all 48 final neural sentence models and all 32 word models, fold-training vocabularies, transferred tensors, fixed histories, per-reader ranks and aggregate metrics. Sentence initialization tensors match the earlier GRU exactly. 2,387 prior protected files remain SHA256-identical; 19,190 dataset/source metadata entries retain size and modification time. Existing local changes, datasets, targets, checkpoints, results, main report, presentation and .venv were preserved. No packages were installed or upgraded.

These are development results on the same eight readers already used in prior experiments. Although no outcomes select epochs or settings in this run, its hypothesis and design were informed by development history. Overlapping folds, repeated words and repeated sentences are not independent cohorts. No significance claim, independent word-level sample size or confidence interval is justified. Previously reused outer validation would not be independent confirmation; neither outer-validation nor test EEG arrays were loaded or evaluated here. No test features were extracted. The task is retrieval among repeated known sentence classes, not unrestricted generation or unseen-sentence decoding.

The source remains gaze/stimulus-assisted and annotation-segmented. Word order and missingness expose structure; word-stage one-hot supervision can learn frequent lexical classes without semantic generalization. This minimal stage uses one short budget, one seed and only a shared projection, so failure does not show that every form of word pretraining fails or that EEG contains no lexical information. Unverified exact producer revisions, source-session preprocessing, gamma2 descriptor/code discrepancy and possible physical boundary tails remain. No text repair, target/candidate/exclusion change or search expansion was made.

## Artifacts and exact commands

The new result directory contains the frozen configuration and implementation hashes; training source/cache/target hashes; exact exported target rows; source pairing evidence; candidate pools; fold identities and vocabularies; all preprocessing and Gaussian moments; class mappings; initial, word-stage and sentence-stage checkpoints; update histories; per-word and per-sentence ranks; all comparisons; decision; access record; replay verification; preservation checks; and command/log records. It contains no outer/test metrics.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_word_pretrain_v1.py tests/test_gru_pilot.py tests/test_gru_retrieval_loss.py -q
.\.venv\Scripts\python.exe scripts/prepare_word_pretrain_v1.py --output results/word_pretrain_v1_20261101
.\.venv\Scripts\python.exe -u scripts/run_word_pretrain_v1.py --output results/word_pretrain_v1_20261101
.\.venv\Scripts\python.exe -u scripts/verify_word_pretrain_v1.py --output results/word_pretrain_v1_20261101
.\.venv\Scripts\python.exe scripts/report_word_pretrain_v1.py --output results/word_pretrain_v1_20261101
```
