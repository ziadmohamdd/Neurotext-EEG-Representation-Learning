# Fixed within-ZuCo1 CNN loss comparison

Six conditions share the same duration-QC cohort, seed-42 subject split,
8-second input recipe, CNN architecture, initialization, optimizer and validation
selection rule. Only the forward text candidate scope changes from the two-item
batch to all 300 unique training sentence embeddings. The reverse batch term
and 1/2 weights remain fixed. Real/Gaussian/shuffled controls are matched.

The comparison does not show reliable retrieval beyond controls. Real Recall@1
stays at two hits out of 592, MRR decreases slightly, and Gaussian MRR remains
higher in both arms. Validation alone prefers the original batch loss for real
EEG. One extra Recall@5 hit or the tiny edge over one shuffled score is not
evidence of success. This is one exploratory seed/split with two test readers.

See [the complete audit and report](../../cnn_candidate_comparison_20261030.md).
`comparison.json` and the CSVs contain actual Recall@1/5/10, MRR, control gaps,
within-control changes and per-subject gaps. `plan.json`,
`selection_frozen.json` and `real_loss_validation_choice.json` record settings
and selection before new test scores were inspected. `runs/` holds copied
evaluation JSON, configs, histories and split manifests. Best/last model and
optimizer checkpoints remain under `results/cnn_candidate_comparison_20261030/`.

`pairing_audit_20261030/` preserves the original source/cache audit, official
source inventories, checksum-verified MATLAB script and stimulus CSV, waveform
clock matches, text variants and pinned local text-embedding reconstruction.
It supports exact cache construction and sampled source-clock provenance, while
retaining uncertainty about all-run preprocessing/annotations. No labels or
preprocessing were corrected mid-experiment.

`archive_manifest.json` records source/copy checksums. `validation.json` records
separate read-only artifact checks. Updated software suite: 145 passed,
2 optional official-weight skips (`EEG_LABRAM_CACHE` unset), 0 deselected.
The real GPU run warned about unset `CUBLAS_WORKSPACE_CONFIG`; no bitwise GPU
reproducibility is claimed. Prior experiments and the new baseline remain intact.
