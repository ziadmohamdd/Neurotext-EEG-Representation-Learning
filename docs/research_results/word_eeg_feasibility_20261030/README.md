# Word/fixation EEG feasibility audit

See the [full report](../../word_eeg_feasibility_20261030.md).

All 12 ZuCo1 and 18 ZuCo2 NR sentence exports were inspected read-only. The evidence supports a separately versioned gaze-assisted sentence-retrieval view, with explicit missingness, boundary and structural-leakage controls. It does not establish reliable EEG retrieval or complete recording-clock validity. No datasets/targets changed and no models trained.

`evidence/summary.json`, `field_census.json`, `sentence_coverage.json` and `derived_details.json` hold measured coverage, shape/state counts and exact all-empty QC identities. `source_inventory.json` pins all 30 source MAT hashes; `channel_schemas.json` and `electrode_pairs.json` preserve channel definitions. The containment evidence includes one fixation extending 39 samples beyond its parent sentence array. `per_subject_counts_and_examples.json` preserves inspection examples and hashes of all full per-subject reports. `archive_manifest.json` verifies source/copy hashes. The original sequential ZuCo2 partial audit is preserved; remaining subjects were audited with four bounded reader processes, with no duplicate subject counted.

Five audit tests passed. All 494 protected prior artifact/input files remained unchanged. Implementation estimate: 5–8 engineering days for a controlled spectral view; 4–7 additional days for a verified raw-fixation path, subject to source availability.

`source_plans/` preserves both original download plans and the exact saved
ZuCo1 split. `provenance_additions.json` records those copies and the existing
MAT-reader/key implementation snapshots. Together with `archive_manifest.json`,
all **37 source/copy hashes** were verified. `validation.json` records the final
coverage/missingness, preservation, duration-unit and waveform-boundary checks.
