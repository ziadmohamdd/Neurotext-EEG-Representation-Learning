# TRT incremental-information comparison — validation only

**Real TRT has a higher selected validation MRR than both comparators. No test features were extracted or evaluated.**

This new comparison reuses the unchanged `zuco1_trt_sentence_mean_pilot_v1` train/validation cache. All three models use 2,245 training observations, 512 validation queries and the same 298 sentence candidates. There are no new exclusions. ZMG/70 remains the one all-missing EEG training observation; all validation observations are retained. Original targets, source data and previous artifacts remain unchanged.

## Exact shared structure and nested models

Structure is the prior count/duration-only control: **14 count/duration summaries plus 48 training-derived word-count basis columns (62 dimensions)**. All raw structural columns, row order and train/validation arrays are identical across conditions. It includes durations, not only the three summary fields used in the previous pilot’s spectral arms. No word text, token identity, sentence/subject ID or gaze coordinates enter any numeric model matrix. Identity metadata is used only for splitting, labels and provenance.

Structural-only selection uses the same mean/median imputation, standard/robust-IQR scaling and seven ridge alphas as the prior pilot (28 settings). Every transform parameter is fitted on training data only; the setting is selected by validation MRR. The selected structural transform is then **frozen and reused byte-for-byte in both combined models**, with structural block weight 1. Its selected alpha is 10, imputation median, scaling standard.

Each combined model appends only the cached **840 TRT amplitude means**, using the real or previously matched Gaussian block. Their original structural tails are discarded. Both receive exactly the same search: EEG mean/median imputation × standard/robust-IQR scaling × alphas `[0.1,1,10,100,1000,10000,100000]` × EEG weights `[0,0.01,0.03,0.1,0.3,1,3]` = **196 settings each**. EEG transforms fit their respective training blocks only. The shared structural transform is not refitted or reselected for either combined model.

The ridge design is `[transformed_structure, weight × transformed_EEG]` (902 dimensions). **Weights apply after scaling**, so the scaler cannot cancel them. A weight of zero exactly reuses the frozen-structural ridge path at each alpha, with zero stored EEG coefficients. All 28 zero-weight entries per combined model match structural-only validation metrics at the same alpha exactly. Selection maximizes validation MRR; ties within 1e-12 prefer the first declared entry, with ascending EEG weight as the outer loop, so zero wins ties. Thus adding a block cannot force a worse selected validation MRR than structure alone. This nesting is an experimental design property, not evidence that the extra block contains useful information.

Gaussian features and their exact word counts/missingness masks are reused unchanged; no new draw or feature extraction occurs. The [pilot report](spectral_trt_pilot_v1_20261030.md) documents training-only Gaussian marginal matching and TRT source provenance. Duration fields remain source sample counts at 500 Hz, not milliseconds. Gaze-assisted segmentation, supplied full-session preprocessing, differing gamma2 documentation and potential sentence-boundary tails remain limitations. No punctuation repair or test-source read was performed.

## Selected validation results

Recall is percent of the same 512 queries against 298 candidates. MRR is not accuracy.

| Model | R@1 % | R@5 % | R@10 % | MRR | Alpha | EEG weight | EEG imputation / scaling |
|---|---:|---:|---:|---:|---:|---:|---|
| structural_only | 17.7734 | 40.6250 | 51.3672 | 0.292012 | 10 | 0 | — |
| structural_real | 17.3828 | 41.0156 | 51.5625 | 0.292310 | 10 | 0.01 | median / standard |
| structural_gaussian | 17.7734 | 40.8203 | 51.1719 | 0.292156 | 10 | 0.01 | mean / robust_iqr |
| Uniform-ranking expectation | 0.3356 | 1.6779 | 3.3557 | 0.021060 | — | — | — |

## Incremental differences

Recall differences are percentage points; MRR differences are absolute.

| Comparison | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |
|---|---:|---:|---:|---:|
| structural_real_minus_structural_only | -0.3906 | +0.3906 | +0.1953 | +0.000299 |
| structural_real_minus_structural_gaussian | -0.3906 | +0.1953 | +0.3906 | +0.000154 |
| structural_gaussian_minus_structural_only | +0.0000 | +0.1953 | -0.1953 | +0.000144 |

## Validation readers

| Model | Reader | Queries | R@1 % | R@5 % | R@10 % | MRR |
|---|---|---:|---:|---:|---:|---:|
| structural_only | zuco1:ZJS | 216 | 17.5926 | 42.1296 | 50.9259 | 0.288718 |
| structural_only | zuco1:ZKW | 296 | 17.9054 | 39.5270 | 51.6892 | 0.294415 |
| structural_real | zuco1:ZJS | 216 | 16.2037 | 42.1296 | 51.8519 | 0.285188 |
| structural_real | zuco1:ZKW | 296 | 18.2432 | 40.2027 | 51.3514 | 0.297507 |
| structural_gaussian | zuco1:ZJS | 216 | 17.5926 | 41.6667 | 50.9259 | 0.289553 |
| structural_gaussian | zuco1:ZKW | 296 | 17.9054 | 40.2027 | 51.3514 | 0.294055 |

Per-reader incremental differences, all selection rows and all 512 query ranks per model are retained in the archive. These two readers form one reused validation split, not replicated cohorts.

## Interpretation and limitations

Real TRT has a higher selected validation MRR than both comparators.

Real MRR is above structural-only: **True**; above Gaussian-combined: **True**; positive gaps to both for each validation reader: **False**.

The selected MRR increases are tiny: real minus structural is **+0.000299**, and real minus Gaussian is **+0.000154**. Real Recall@1 decreases by two hits relative to both comparators. For ZJS, real MRR is lower than both; for ZKW it is higher. This does not demonstrate a consistent useful increment from real TRT on the available validation readers.

These are selected validation point estimates. Structural-only has 28 searched settings; each combined model has 196, making superiority over structural-only alone susceptible to search optimism. Real and Gaussian combined searches are exactly matched, but this validation split was already used in earlier investigations. No independent significance or generalization claim follows from a small selected increase, training fit or a single metric. The zero-EEG option and observed Gaussian gain must inform whether a real-feature increase is useful. No grid expansion, extra seed or test evaluation followed the results.

## Reproduction and preservation

Four focused tests passed in 1.71 s. They verify exact zero-weight nesting, effective post-scaling weights, reference-ridge predictions, training-only preprocessing and identical search candidates. The run replays every selected saved model to reproduce its validation ranks, checks all zero-weight metrics against structural-only and hashes the raw/transformed structural arrays. All **585 protected prior files** remained unchanged. The feature cache and targets were reused byte-for-byte; no MAT or raw EEG data were opened. No test feature matrix was loaded, extracted or evaluated.

Results: `results\trt_incremental_v1_20261030`. The [archive](research_results/trt_incremental_v1_20261030/README.md) contains the frozen plan, preprocessing provenance, complete grids, metrics, ranks, differences and source snapshots. Coefficients and full preprocessing arrays remain in the new result folder.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_incremental_ridge.py -q
.\.venv\Scripts\python.exe -u scripts/run_trt_incremental_comparison.py --output results/trt_incremental_v1_20261030
```

The runner refuses existing result/report/archive destinations. Use new versioned destinations for reproduction.

An initial relative-path error interrupted archive packaging after all three
fits, grids and verification results were saved. Packaging was completed with
`--finalize-only`, without refitting or altering any selected model, rank or
score. The original fit-time source snapshot remains intact; the corrected
packaging source and its hash are in `packaging/` and
`packaging_provenance.json` in the archive.
