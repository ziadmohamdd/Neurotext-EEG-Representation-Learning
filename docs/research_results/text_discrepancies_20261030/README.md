# Text discrepancy investigation

See the [full source-based investigation](../../text_discrepancy_investigation_20261030.md).

Checked-in source snapshots and final audit copies are in [`artifacts/`](artifacts/).
[`archive_manifest.json`](artifacts/archive_manifest.json) records 19 source/copy
hash matches; [`validation.json`](artifacts/validation.json) records separate
count, input-hash, local-link and preservation checks. No EEG arrays or model
checkpoints were duplicated into this archive.

Original investigation outputs are in
[`results/text_discrepancy_investigation_20261030/analysis_final/`](../../../results/text_discrepancy_investigation_20261030/analysis_final/).
`summary.json` contains per-split counts and input/code hashes;
`discrepancies.csv` and `.json` classify all 37 sentence rows with exact source
texts and indices; `affected_observations.json` enumerates 408 prepared and
406 duration-QC retained affected observations. `source_provenance.json` pins
official file versions and hashes, and `preservation.json` verifies 415 prior
result/input files remained unchanged.

There are 21 formatting-only, 8 orthographic/canonical-key, 8 substantive and
0 unresolved row mappings. The separate annotation CSV maps to the same ZuCo1
normal-reading subset but is not an established corrected export release.
The source sentence MAT exactly matches all 300 exported targets. Keep existing
targets/results. A separately versioned, three-sentence punctuation repair is
proposed in the report and has not been applied. Five audit-only tests passed;
no training or new model evaluation ran.
