# Full-word-sequence GRU pilot v1 — training-reader CV

**One bounded pilot completed. Training-only descriptive gate: failed; stopped without outer validation.**

The real sequence encoder did not establish a useful, consistent advantage beyond the matched controls and structural comparator.

## Training-reader holdout results

These are equal-reader macro training-CV results on the original eight training readers, 2,245 observations and unchanged 300-candidate pool. Recall is percent; MRR is not accuracy. The common epoch was selected from 4/8/12 using real-EEG training-CV Recall@1, with ties preferring the earlier epoch. Selected epoch: **8**. Every neural condition uses that same epoch.

| Condition | R@1 % | R@5 % | R@10 % | MRR |
|---|---:|---:|---:|---:|
| real_eeg | 0.5424 | 1.7590 | 3.4716 | 0.023048 |
| mean_eeg | 0.4011 | 1.7373 | 3.5249 | 0.022454 |
| matched_gaussian | 0.4043 | 1.6910 | 3.5333 | 0.021327 |
| class_shuffled | 0.3578 | 1.8316 | 3.3172 | 0.021432 |
| word_order_shuffled | 0.4412 | 1.7503 | 3.5206 | 0.022068 |
| missingness_only | 0.3578 | 1.6588 | 3.3200 | 0.021191 |
| structural_only | 16.1129 | 40.2867 | 52.0525 | 0.280271 |

## Real EEG minus comparators

| Comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Gate |
|---|---:|---:|---:|---:|---:|---|
| mean_eeg | +0.1413 | +0.0217 | -0.0533 | +0.000594 | 3/8 | False |
| matched_gaussian | +0.1382 | +0.0679 | -0.0617 | +0.001721 | 3/8 | False |
| class_shuffled | +0.1846 | -0.0726 | +0.1544 | +0.001616 | 5/8 | False |
| word_order_shuffled | +0.1012 | +0.0086 | -0.0490 | +0.000980 | 1/8 | False |
| missingness_only | +0.1846 | +0.1002 | +0.1516 | +0.001857 | 5/8 | False |
| structural_only | -15.5705 | -38.5277 | -48.5809 | -0.257222 | 0/8 | False |

The prespecified gate requires real EEG to exceed every comparator by at least 2 percentage points Recall@1 and 0.005 MRR, with positive Recall@1 differences on at least six of eight readers for every comparison. It is a descriptive stopping rule, not a significance test.

## Architecture, inputs and controls

The model has 45,952 trainable parameters: an 841→32 linear projection (26,944), tanh, one unidirectional 32-unit GRU (6,336), and a 32→384 sentence-embedding head (12,672). It consumes all source word positions with no binning, truncation or timing assumptions. Source word order is stimulus order, not chronological fixation order.

The 840 EEG coordinates undergo signed log1p and within-word centering separately over 105 channels in each of eight bands. Mean/SD are fitted only on complete words from fold-training readers. Missing words are imputed to that training mean, producing scaled zeros, and an extra observed-word indicator is appended. Missing slots remain recurrent steps. Batch padding is separate: packed sequences use true word counts, so padded tails never reach the GRU. All-missing ZMG/70 remains as its full-length zero/indicator-zero sequence; predictions may reflect length and learned biases. A hypothetical zero-length sequence uses one zero fallback step. No word text, word ID, sentence ID, subject ID or eye-tracking measurement enters a neural input.

Matched Gaussian draws preserve exact word counts and every missing coordinate. Their marginal moments are fitted on raw fold-training words only, before the shared nonlinear transform; Gaussian scaling is independently fitted on Gaussian fold-training words. This null does not preserve the full real-feature distribution or covariance. Word-order shuffling permutes complete vectors only among observed slots, preserving missing positions and sentence means. The mean comparator repeats each sentence’s mean scaled EEG vector at its observed slots, preserving count and mask. Missingness-only zeros all 840 amplitude coordinates while retaining the same indicator and true sequence length. Class shuffling consistently deranges training sentence classes across repeated observations; held-out labels remain true.

All six neural conditions use identical architecture, seed-42 initial weights, minibatch order, AdamW (learning rate 0.001, weight decay 0.01), cosine embedding loss, batch size 64, gradient clipping 1, and 12 epochs with checkpoints at 4/8/12. No learning-rate, width or optimizer search was performed. The existing structural comparator uses the same observations and targets, its original 14 count/duration summaries plus a fold-training count basis, training-fitted median imputation/standardization and fixed ridge alpha 1. That alpha came from preceding training-only semantic CV; no new structural grid was searched. Duration summaries enter only this comparator.

## Per-reader training-CV metrics

| Condition | Reader | R@1 % | R@5 % | R@10 % | MRR |
|---|---|---:|---:|---:|---:|
| real_eeg | ZDM | 0.3460 | 0.6920 | 2.7682 | 0.018666 |
| real_eeg | ZDN | 0.6734 | 1.3468 | 2.6936 | 0.022205 |
| real_eeg | ZGW | 0.7117 | 1.7794 | 3.9146 | 0.026099 |
| real_eeg | ZJM | 0.7782 | 2.3346 | 5.4475 | 0.026249 |
| real_eeg | ZJN | 0.6803 | 1.7007 | 3.7415 | 0.023460 |
| real_eeg | ZKH | 0.3401 | 2.0408 | 3.0612 | 0.022512 |
| real_eeg | ZMG | 0.0000 | 1.7483 | 2.0979 | 0.016686 |
| real_eeg | ZPH | 0.8097 | 2.4291 | 4.0486 | 0.028510 |
| mean_eeg | ZDM | 0.3460 | 1.3841 | 4.1522 | 0.020717 |
| mean_eeg | ZDN | 0.3367 | 1.6835 | 4.0404 | 0.023434 |
| mean_eeg | ZGW | 0.7117 | 2.1352 | 3.9146 | 0.025903 |
| mean_eeg | ZJM | 0.3891 | 1.9455 | 4.6693 | 0.024813 |
| mean_eeg | ZJN | 0.6803 | 2.3810 | 2.7211 | 0.022793 |
| mean_eeg | ZKH | 0.3401 | 1.7007 | 2.7211 | 0.021885 |
| mean_eeg | ZMG | 0.0000 | 1.0490 | 3.1469 | 0.018190 |
| mean_eeg | ZPH | 0.4049 | 1.6194 | 2.8340 | 0.021899 |
| matched_gaussian | ZDM | 0.3460 | 1.3841 | 4.4983 | 0.021064 |
| matched_gaussian | ZDN | 0.3367 | 1.6835 | 2.6936 | 0.019017 |
| matched_gaussian | ZGW | 0.7117 | 2.1352 | 3.9146 | 0.024885 |
| matched_gaussian | ZJM | 0.0000 | 1.5564 | 3.5019 | 0.019248 |
| matched_gaussian | ZJN | 0.3401 | 1.0204 | 2.7211 | 0.018604 |
| matched_gaussian | ZKH | 0.3401 | 2.3810 | 3.7415 | 0.021412 |
| matched_gaussian | ZMG | 0.3497 | 1.7483 | 3.1469 | 0.021745 |
| matched_gaussian | ZPH | 0.8097 | 1.6194 | 4.0486 | 0.024644 |
| class_shuffled | ZDM | 0.3460 | 1.3841 | 2.4221 | 0.020592 |
| class_shuffled | ZDN | 0.3367 | 1.6835 | 3.3670 | 0.020299 |
| class_shuffled | ZGW | 0.3559 | 2.1352 | 3.2028 | 0.021779 |
| class_shuffled | ZJM | 0.3891 | 1.9455 | 3.5019 | 0.022540 |
| class_shuffled | ZJN | 0.3401 | 1.7007 | 3.4014 | 0.021484 |
| class_shuffled | ZKH | 0.3401 | 2.3810 | 3.7415 | 0.021555 |
| class_shuffled | ZMG | 0.3497 | 1.3986 | 2.4476 | 0.018881 |
| class_shuffled | ZPH | 0.4049 | 2.0243 | 4.4534 | 0.024330 |
| word_order_shuffled | ZDM | 0.3460 | 1.0381 | 2.7682 | 0.019082 |
| word_order_shuffled | ZDN | 0.6734 | 2.0202 | 3.3670 | 0.023975 |
| word_order_shuffled | ZGW | 0.7117 | 2.1352 | 3.5587 | 0.025992 |
| word_order_shuffled | ZJM | 0.7782 | 2.3346 | 5.4475 | 0.027884 |
| word_order_shuffled | ZJN | 0.6803 | 1.7007 | 2.7211 | 0.021384 |
| word_order_shuffled | ZKH | 0.3401 | 1.7007 | 3.4014 | 0.020913 |
| word_order_shuffled | ZMG | 0.0000 | 1.0490 | 2.4476 | 0.015330 |
| word_order_shuffled | ZPH | 0.0000 | 2.0243 | 4.4534 | 0.021986 |
| missingness_only | ZDM | 0.3460 | 1.7301 | 2.4221 | 0.020249 |
| missingness_only | ZDN | 0.3367 | 1.3468 | 3.0303 | 0.020167 |
| missingness_only | ZGW | 0.3559 | 1.4235 | 3.2028 | 0.021393 |
| missingness_only | ZJM | 0.3891 | 1.9455 | 3.5019 | 0.022205 |
| missingness_only | ZJN | 0.3401 | 1.7007 | 3.4014 | 0.020635 |
| missingness_only | ZKH | 0.3401 | 1.7007 | 3.4014 | 0.021279 |
| missingness_only | ZMG | 0.3497 | 1.3986 | 3.1469 | 0.019656 |
| missingness_only | ZPH | 0.4049 | 2.0243 | 4.4534 | 0.023944 |
| structural_only | ZDM | 14.8789 | 40.1384 | 51.9031 | 0.270690 |
| structural_only | ZDN | 12.7946 | 39.7306 | 52.1886 | 0.254211 |
| structural_only | ZGW | 18.1495 | 39.5018 | 51.6014 | 0.290449 |
| structural_only | ZJM | 15.5642 | 39.6887 | 51.3619 | 0.275772 |
| structural_only | ZJN | 17.3469 | 37.0748 | 50.0000 | 0.277953 |
| structural_only | ZKH | 16.3265 | 40.4762 | 52.0408 | 0.285932 |
| structural_only | ZMG | 16.4336 | 41.9580 | 51.0490 | 0.285010 |
| structural_only | ZPH | 17.4089 | 43.7247 | 56.2753 | 0.302149 |

## Outer validation and test status

Outer validation was not evaluated and its EEG cache members were not loaded because the training-only gate failed. No outer-validation scores are reported.

No raw source MAT file was opened, no test feature array was loaded, and no test extraction or evaluation was performed. The existing 512-query/298-candidate outer cohort and all targets/exclusions remain unchanged.

## Verification, preservation and limitations

Before fitting, 16 focused tests passed (5 new GRU tests and 11 existing word-order/spectral tests). They cover missingness, all-missing fallback, padding invariance, training-only transforms and moments, matched controls, class-consistent shuffling and reader separation. The existing .venv was used unchanged (Python 3.11.9, PyTorch 2.7.1+cu118, RTX 3050 Laptop GPU). The sandbox could not resolve its base interpreter; execution outside the sandbox succeeded without installation or upgrades. 1708 pre-existing files remain SHA256-identical; 19190 dataset files retain size and modification time. Raw/test sources were inventoried through filesystem metadata only. Existing datasets, targets, checkpoints, results, report and local changes were preserved.

These CV scores reuse the training readers both for duration selection and description and are optimistic development results. Prior experiments have already informed the representation and protocol. Outer validation has also been reused historically and would not provide independent confirmation. Repeated sentences and overlapping folds are not independent cohorts; no confidence interval or significance claim is made. Retrieval of known candidate sentences is not unrestricted text generation or unseen-sentence decoding. Gaze/stimulus-assisted segmentation, missingness and length can reveal annotation structure. Producer preprocessing, exact revision, gamma2 descriptor/code discrepancy and physical boundary limitations from the source audits remain. One small architecture, one seed and a short budget cannot establish that all sequence models fail or that EEG contains no semantic information. No search was expanded after this result.

## Artifacts and commands

The result directory contains the frozen plan/configuration, source/cache and implementation hashes, candidate pools, every fold identity and class mapping, train-only preprocessing and Gaussian moments, identical initial weights, checkpoints at every prespecified epoch, training histories, per-query ranks, all scored epochs, shared epoch selection, per-reader metrics, control differences, decision, access record and preservation checks. No previous report or result was replaced.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe -m pytest tests/test_gru_pilot.py tests/test_word_order.py tests/test_spectral_pilot.py -q
.\.venv\Scripts\python.exe scripts/run_gru_pilot.py --output results/gru_pilot_v1_20261101
.\.venv\Scripts\python.exe scripts/report_gru_pilot.py
```
