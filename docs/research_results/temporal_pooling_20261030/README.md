# Validation-only temporal pooling comparison

See the [full report](../../temporal_pooling_comparison_20261030.md).

**No useful validation improvement relative to controls. Stopped without test evaluation.**

`plan.json` fixes the recipe, initialization, validation decision rule and test prohibition before fitting. `validation_summary.json`, `validation.csv`, condition-level `validation.json` and `control_gaps.json` contain only validation measurements. `decision.json` records the screen. Full checkpoints remain in the new results directory. Original targets, including the proposed punctuation repairs, were unchanged.

The variant has 80,032 parameters versus 67,744, with exact shared-layer initialization. All six fits used the original batch contrastive loss, saved seed-42 splits and common training budget. No test retrieval was performed. `validation_checks.json` and `archive_manifest.json` preserve checksums and separate verification outcomes.
