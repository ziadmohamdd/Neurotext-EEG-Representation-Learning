# Global-average versus four-segment CNN pooling — validation only

**No useful validation improvement relative to controls. Stopped without test evaluation.**

This fixed, exploratory seed-42 comparison changes only temporal pooling and the input width of its linear head. All six real/Gaussian/shuffled fits used the existing export-faithful ZuCo1 targets. The proposed punctuation repair was not applied. Previous results, checkpoints and source data remain intact.

## Fixed design and initialization

The saved subject split is unchanged: 2,245 training observations from ZDM/ZDN/ZGW/ZJM/ZJN/ZKH/ZMG/ZPH; 512 validation observations from ZJS/ZKW; the 592 test observations from ZAB/ZKB were not evaluated. The existing duration-QC cohort, 105-channel arrays at 200 Hz, training-fitted channel normalizer and deterministic center crop of at most 1,600 samples (8 seconds) are unchanged. No additional signal filtering was introduced. The existing cached 300 MiniLM sentence embeddings are reused byte-for-byte. These sentence classes recur across training and validation readers: this is held-out-reader retrieval of seen sentence classes, not unseen-sentence generalization.

The loss remains symmetric multi-positive **batch** contrastive cross entropy with temperature 0.07, the original validation-selected loss. Each microbatch contains two distinct sentences; accumulation over eight microbatches does not create a 16-candidate contrastive denominator. Both architectures use AdamW (LR 0.001, weight decay 0.01), gradient clipping 1, the same seed and data/control generators, fp16-if-supported, a maximum of 50 epochs and patience 8. Each best epoch is selected solely by minimum held-out validation batch CE. Full-pool validation retrieval is computed only after that checkpoint is frozen; it never selects an epoch. Actual early-stopped durations are reported below, rather than claimed to be equal.

Global averaging pools each channel to one value. The variant uses `AdaptiveAvgPool1d(4)`, concatenates the four ordered bins in channel-major order, and expands the encoder head from 32→128 to 128→128. For lengths not divisible by four, adaptive pooling can overlap boundary samples; it does not discard or pad the valid signal. These are relative temporal bins of the retained crop, not word-aligned segments. Each valid sequence is encoded separately so batch padding never enters convolution, GroupNorm or pooling.

| Trainable parameters | Global | Four segments |
|---|---:|---:|
| Convolutional features and GroupNorm | 13,216 | 13,216 |
| Encoder linear head including bias | 4,224 | 16,512 |
| Text projection and LayerNorm | 50,304 | 50,304 |
| **Total** | **67,744** | **80,032** |

The variant adds 12,288 parameters (18.14%). Every shared tensor is initialized identically, including the convolutional stack, head bias and text projection/LayerNorm. Each of the four new head columns for a channel receives its original weight divided by four (`repeat_interleave`). This makes the initial outputs equal within numerical tolerance for equal-sized bins while allowing bin-specific gradients. For uneven bins, equal weighting of adaptive-bin means need not equal the exact global mean. The global initialization hash matches the preserved earlier CNN baseline; all three controls use the same initialization per architecture. Exact hashes and shared tensor names are in `plan.json`.

Gaussian input is deterministic per observation, zero-mean in expectation and scaled by each real cropped channel’s standard deviation before the same normalizer. It retains amplitude/length structure and is not a proof of complete absence of dataset information. Shuffled control deranges training sentence classes consistently across repeated observations; its validation inputs and target keys remain real/correct. Controls and draws are identical between architectures.

## Validation measurements

Recall values below are percentages of 512 validation queries against the same 298 unique sentence candidates. MRR is reported as reciprocal rank, never converted into accuracy. Epoch numbers in the table are one-based.

| Pooling | Condition | Recall@1 % | Recall@5 % | Recall@10 % | MRR | Best epoch | Epochs run | Val batch CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| global | real | 0.195 | 1.562 | 3.125 | 0.018858 | 1 | 9 | 0.697376 |
| global | gaussian | 0.195 | 0.977 | 2.930 | 0.018387 | 3 | 11 | 0.693672 |
| global | shuffled | 0.195 | 2.344 | 3.516 | 0.020681 | 2 | 10 | 0.694353 |
| four_segment | real | 0.195 | 1.562 | 2.734 | 0.019086 | 1 | 9 | 0.697719 |
| four_segment | gaussian | 0.391 | 1.562 | 2.539 | 0.021347 | 3 | 11 | 0.693726 |
| four_segment | shuffled | 0.195 | 1.172 | 3.125 | 0.018683 | 1 | 9 | 0.696531 |
| Uniform random ranking | analytic expectation | 0.336 | 1.678 | 3.356 | 0.021060 | — | — | — |

Chance is an analytic uniform-ranking reference, not an additional trained control. Batch CE is a two-way objective near log(2); it must not be interpreted as 298-way retrieval accuracy.

### Real-minus-control gaps

Recall gaps are percentage points; MRR gaps are absolute reciprocal-rank differences.

| Pooling | Comparison | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |
|---|---|---:|---:|---:|---:|
| global | real_minus_gaussian | +0.000 | +0.586 | +0.195 | +0.000470 |
| global | real_minus_shuffled | +0.000 | -0.781 | -0.391 | -0.001823 |
| four_segment | real_minus_gaussian | -0.195 | +0.000 | +0.195 | -0.002261 |
| four_segment | real_minus_shuffled | +0.000 | +0.391 | -0.391 | +0.000403 |

### Four-segment minus global changes

| Condition or control gap | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |
|---|---:|---:|---:|---:|
| real | +0.000 | +0.000 | -0.391 | +0.000228 |
| gaussian | +0.195 | +0.586 | -0.391 | +0.002959 |
| shuffled | +0.000 | -1.172 | -0.391 | -0.001998 |
| real_minus_gaussian | -0.195 | -0.586 | +0.000 | -0.002731 |
| real_minus_shuffled | +0.000 | +1.172 | +0.000 | +0.002227 |

### Validation participants

These two readers form one validation split, not independent replicated cohorts.

| Pooling | Condition | Participant | Queries | R@1 % | R@5 % | R@10 % | MRR |
|---|---|---|---:|---:|---:|---:|---:|
| global | real | zuco1:ZJS | 216 | 0.000 | 0.926 | 3.241 | 0.015545 |
| global | real | zuco1:ZKW | 296 | 0.338 | 2.027 | 3.041 | 0.021275 |
| global | gaussian | zuco1:ZJS | 216 | 0.000 | 0.000 | 1.852 | 0.014096 |
| global | gaussian | zuco1:ZKW | 296 | 0.338 | 1.689 | 3.716 | 0.021519 |
| global | shuffled | zuco1:ZJS | 216 | 0.463 | 1.389 | 2.315 | 0.018077 |
| global | shuffled | zuco1:ZKW | 296 | 0.000 | 3.041 | 4.392 | 0.022581 |
| four_segment | real | zuco1:ZJS | 216 | 0.000 | 0.926 | 2.778 | 0.016921 |
| four_segment | real | zuco1:ZKW | 296 | 0.338 | 2.027 | 2.703 | 0.020666 |
| four_segment | gaussian | zuco1:ZJS | 216 | 0.463 | 1.389 | 1.389 | 0.019619 |
| four_segment | gaussian | zuco1:ZKW | 296 | 0.338 | 1.689 | 3.378 | 0.022607 |
| four_segment | shuffled | zuco1:ZJS | 216 | 0.000 | 0.926 | 2.778 | 0.015883 |
| four_segment | shuffled | zuco1:ZKW | 296 | 0.338 | 1.351 | 3.378 | 0.020726 |

## Decision and limitations

Before the fits, the descriptive validation screen required higher real MRR, positive and improved real-minus-Gaussian and real-minus-shuffled MRR gaps, positive four-segment real MRR gaps for each validation participant, and at least two recall cutoffs above both controls while no worse than the global real model. A 1e-12 roundoff tolerance is used for gap improvements. This is a prespecified decision screen, not a significance threshold or a claim of clinical/scientific validity. No settings, seed, loss, epoch rule or preprocessing were changed after seeing validation retrieval.

- `real_mrr_improves`: passed.
- `real_beats_gaussian_mrr`: failed.
- `mrr_gap_improves_vs_gaussian`: failed.
- `zuco1:ZJS_real_beats_gaussian_mrr`: failed.
- `zuco1:ZKW_real_beats_gaussian_mrr`: failed.
- `real_beats_shuffled_mrr`: passed.
- `mrr_gap_improves_vs_shuffled`: passed.
- `zuco1:ZJS_real_beats_shuffled_mrr`: passed.
- `zuco1:ZKW_real_beats_shuffled_mrr`: failed.
- `recall_corroboration`: failed.

**No useful validation improvement relative to controls. Stopped without test evaluation.**

One previously used seed/split and two validation readers cannot establish reliable EEG retrieval. Validation reuse across investigations can itself cause selection optimism. An isolated recall increase, training memorization, or a higher real score accompanied by comparable control gains is insufficient. Extra linear head capacity is confounded with temporal pooling in this comparison; the result cannot isolate a pure temporal-order mechanism. No additional seeds, hyperparameter trials, target repairs or test runs were launched.

The GPU emitted the same unset `CUBLAS_WORKSPACE_CONFIG` deterministic-operation warning as the earlier experiment, plus an `adaptive_avg_pool2d_backward_cuda` nondeterministic-implementation warning for the four-bin arm. Deterministic seeding was retained with warn-only behavior, but bitwise GPU reproducibility is not assumed; this single draw does not measure run-to-run numerical variation. The archive records direct model-tensor/validation-loss reproduction checks for the three global fits.

## Artifacts and validation

All old result/input files checked by the runner remained unchanged (**436 files**). The manifest and text-cache SHA-256 values match the prior study. Model checkpoints remain only under `results/temporal_pooling_20261030/`; no prior checkpoint was overwritten. The new [archive](research_results/temporal_pooling_20261030/README.md) contains measured validation summaries, per-query ranks, participant results, candidate pools, split/normalizer/configuration provenance, history and source snapshots. `archive_manifest.json` records source/copy hashes, and `validation_checks.json` records separate artifact checks.

Focused software validation: **34 passed in 5.50 s**, no skips, for `tests/test_temporal_pooling.py`, `tests/test_models.py` and `tests/test_candidate_loss.py`. Checks cover baseline compatibility, matched initialization, parameter counts, padding/short sequences, segment ordering/gradients, unchanged loss and control-relative screening. Two additional report bookkeeping tests passed for exact validation-pool matching and rejection of test/different candidate pools. These checks caught and corrected an initial report-only assumption of 300 validation candidates: the measured evaluation pool has 298; the training bank/cache has 300. All six evaluations used 298 from the outset, so no ranks or model outputs were changed. This is not a rerun of the previously recorded full suite.

Commands executed with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_temporal_pooling.py tests/test_models.py tests/test_candidate_loss.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_pooling_report.py -q
.\.venv\Scripts\python.exe -u scripts/run_temporal_pooling_comparison.py --output results/temporal_pooling_20261030
.\.venv\Scripts\python.exe scripts/report_temporal_pooling_comparison.py --run results/temporal_pooling_20261030
```

Runner and reporter refuse existing output destinations. To reproduce, use fresh versioned paths.
