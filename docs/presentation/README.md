# Consolidated presentation

Latest deck: [35-slide PowerPoint](NeuroText_consolidated_v2_20261105.pptx) and
[PDF preview](NeuroText_consolidated_v2_20261105.pdf). Its slides 1–26 are byte-identical
to the earlier deck. Slides 27–35 add the later pilots, separate word-task scores,
expected N-way accuracy, a project-brief checklist and the current conclusion, all
from saved evidence. `scripts/consolidate_later_evidence.py` builds it, and its
[manifest](../research_results/presentation_update_20261105/manifest.json) records
the source hashes and checks. The sections below describe slides 1–26.

Earlier deck, unchanged: [26-slide PowerPoint](NeuroText_consolidated.pptx) and
[PDF preview](NeuroText_consolidated.pdf). It stops at the 2026-10-30 spectral and
incremental comparisons. Slides 13–20 add source-only selection,
all 18 wider-grid seed-level results, descriptive means/SD, paired controls,
the separate original-grid seed-42 comparison, provenance and recorded validation.
Tables and text are editable. Figures use archived scientific PNGs.
The PDF was exported from the final deck by local Microsoft PowerPoint.

Slides 21–26 consolidate the **validation-only** TRT spectral pilot and
incremental-information comparison. The shared cohort has 2,245 training
observations, 512 validation queries and 298 candidates. Structural
count/duration performance (MRR 0.292012) is reported separately from the
absence of a consistent useful incremental EEG advantage.

The pilot searches 28 settings per arm. Each incremental combined model
searches **196 settings**, including zero EEG weight, using identical structural
inputs and a frozen training-fitted transform. Validation is repeatedly reused
across these and earlier investigations. The selected scores are exploratory,
not independent held-out estimates. Real-minus-structure MRR is +0.000299 and
real-minus-Gaussian-combined MRR is +0.000154, with opposite reader effects and
two fewer Recall@1 hits. **Further tuning and test evaluation are stopped.**
No new fitting, feature extraction or model evaluation ran during consolidation.

The preceding 20-slide deck, PDF, main report, README, outline and continuation
instructions are retained in [archive/pre_spectral_20261030](archive/pre_spectral_20261030/).
The [new evidence archive](../research_results/spectral_validation_consolidation_20261030/README.md)
preserves full-precision stored results. Standalone spectral and incremental
experiment files and feature caches are unchanged. Their targeted checks
(7 and 4 passed) are not a new cumulative full-suite total.

These configurations do not demonstrate reliable retrieval beyond controls.
Cross-release participant independence remains unresolved. All seeds reuse the
same target observations; repeated splits are not independent cohorts. The 50
shared source/target texts remain in inclusive pools. Source-validation MRR
alone selects alpha.

The original 79-report export, six original-grid seed-42 transfer comparisons,
and 18 wider-grid runs across seeds 42/43/44 are distinct sets. Do not pool
grids or describe these results as verified unseen-person transfer.

Validation recorded for the wider-grid consolidation is **139 passed, 2 optional
skips, 0 deselected**. This is a historical count, not the current suite total.
`EEG_LABRAM_CACHE` was unset, so official-weight integration (no network download
during unit tests) and extraction on synthetic fixtures were skipped. Exact
node IDs and reasons are in [the report](../research_results/REPORT.md). The
earlier 513 MiB MATLAB exclusion no longer applies: that test passed. No training,
fitting or model evaluations ran during consolidation.

Original deliverables remain unchanged as `NeuroText_draft.pptx/.pdf` and in
[archive/pre_wider_grid](archive/pre_wider_grid/). That archive also retains the
previous main report, outline, presentation README and legacy builder. Full
summaries, candidate pools, exact configs, copied evaluation JSONs and checksums
are in [the transfer archive](../research_results/wider_grid_transfer/archive_manifest.json).

To create a fresh revision without EEG processing:

```powershell
.\.venv\Scripts\python.exe scripts/consolidate_presentation.py --output docs/presentation/NeuroText_revision.pptx
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/render_presentation.ps1 -InputDeck docs/presentation/NeuroText_revision.pptx -OutputDir results/NEW_RENDER_FOLDER
```

The XML builder preserves the original template and refuses to overwrite outputs.
The renderer requires local Microsoft PowerPoint and writes slide PNGs and a PDF.
Keep manual edits separately. The legacy `scripts/build_presentation.py` creates
the historical 14-slide deck and does not consolidate wider-grid transfer.

The command above rebuilds the historical 20-slide wider-grid revision. The
latest spectral consolidation uses `scripts/consolidate_spectral_validation.py`
and `scripts/finalize_spectral_deck.py`, then `scripts/make_spectral_final_titles.py`, with the archived 20-slide source and
stored validation JSON. These scripts have guarded destinations and must use
fresh versioned paths for another revision. They launch no experiments.
