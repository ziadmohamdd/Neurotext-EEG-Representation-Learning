# TRT spectral pilot v1 — seed-42 validation only

See the [full report](../../spectral_trt_pilot_v1_20261030.md).

**No useful validation advantage over the strongest control. Stopped without test extraction or evaluation.**

All six conditions share 2,245 training observations, 512 validation observations and 298 validation candidates. `plan.json` freezes the TRT family, transformations, candidate pool and validation screen. `feature_view.json` and `feature_view_metadata/` record the new numeric cache, exact Gaussian masks and feature-column whitelist. Word text/identities are not model inputs. Duration controls use source sample counts at 500 Hz.

`validation_summary.json`, `validation.csv`, `real_minus_controls.json` and condition folders contain measured validation results, per-reader metrics, ranks and every selection setting. `decision.json` records the no-test decision. `validation_checks.json` and `archive_manifest.json` record preservation and source/copy hashes. No test features were extracted and no previous target, dataset, checkpoint or experiment result was changed.

`provenance_additions.json` records original exclusion/split files and the
dataset/RNG/normalizer code. `target_cache_metadata.json` pins the reused
sentence-embedding model and revision. Across both copy manifests, 41 hashes
were verified. `final_validation.json` confirms that saved coefficients and
preprocessing reproduce all six validation rank arrays, and that no test cache
or test-subject source extraction occurred.
