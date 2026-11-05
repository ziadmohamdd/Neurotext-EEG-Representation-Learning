# Incremental TRT information — validation only

See the [full report](../../trt_incremental_v1_20261030.md).

Real TRT has a higher selected validation MRR than both comparators.

All three models share the full 62-feature structural block and its frozen train-fitted transform. Real/Gaussian combined models use identical alpha, EEG-preprocessing and post-scaling block-weight searches, including an exact zero-EEG candidate. Both use the original matched Gaussian and real TRT cache. All 2,245 training observations, 512 validation observations and 298 candidates are unchanged. No test features were extracted or evaluated.

`plan.json`, `structural_preprocessing_frozen.json`, `validation_checks.json`, `differences.json`, `validation_summary.json`, condition grids/ranks, source snapshots and `archive_manifest.json` record configuration and verification. Full coefficients remain in the new results directory.

Real's selected MRR edge is +0.000299 over structural-only and +0.000154 over
structural+Gaussian, with lower Recall@1 and opposite MRR directions across
readers. This does not demonstrate a consistent useful increment.

An archive path error after fitting was repaired via `--finalize-only`. No fits
were repeated or scores changed. `packaging_provenance.json` identifies the
corrected packaging code; the original fit-time source snapshot is preserved.
