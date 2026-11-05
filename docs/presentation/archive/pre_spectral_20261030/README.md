# Consolidated presentation

Current deliverables: [20-slide PowerPoint](NeuroText_consolidated.pptx) and
[PDF preview](NeuroText_consolidated.pdf). Slides 13–20 add source-only selection,
all 18 wider-grid seed-level results, descriptive means/SD, paired controls,
the separate original-grid seed-42 comparison, provenance and recorded validation.
Tables and text are editable. Figures use archived scientific PNGs.
The PDF was exported from the final deck by local Microsoft PowerPoint.

These configurations do not demonstrate reliable retrieval beyond controls.
Cross-release participant independence remains unresolved. All seeds reuse the
same target observations; repeated splits are not independent cohorts. The 50
shared source/target texts remain in inclusive pools. Source-validation MRR
alone selects alpha.

The original 79-report export, six original-grid seed-42 transfer comparisons,
and 18 wider-grid runs across seeds 42/43/44 are distinct sets. Do not pool
grids or describe these results as verified unseen-person transfer.

Recorded validation is **139 passed, 2 optional skips, 0 deselected**.
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
