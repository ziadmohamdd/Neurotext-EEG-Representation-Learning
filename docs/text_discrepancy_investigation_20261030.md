# ZuCo1 export versus annotation text investigation — 2026-10-30

The 37 differences are **21 formatting-only, 8 orthographic/canonical-key,
8 substantive wording, and 0 unresolved row mappings**. They are not evidence
of 37 incorrectly paired EEG trials. The 300 exported targets match the source
sentence MAT exactly. The annotation CSV describes the same ZuCo1 normal-reading
material and its rows can be mapped, but it is a different text representation,
not an established replacement target release. **Keep the existing dataset and
results unchanged.** A separate, limited punctuation-decoding repair is proposed
below; it has not been applied. No training, model evaluation, text embedding
generation, or signal preprocessing ran during this investigation.

## Source evidence and release identity

All primary files are from the [ZuCo corpus OSF project](https://osf.io/q3zws/).
The downloaded bytes were checked against the official OSF SHA-256 metadata.
Full hashes, file IDs, versions, dates and download URLs are in
[`source_provenance.json`](../results/text_discrepancy_investigation_20261030/analysis_final/source_provenance.json).
The source snapshots are in the same investigation folder or the preserved
`results/pairing_audit_20261030/` folder.

| Primary source | OSF path / role | Recorded version and modification |
|---|---|---|
| [relations_labels_task2.csv](https://osf.io/download/tu6yx/) | `/task_materials/`; 300 rows with relation labels | v1, 2018-05-14 |
| [relations_normal_reading.csv](https://osf.io/download/xgbr5/) | `/task2 - NR/Preprocessed/`; 595 rows, including practice and unused trailing material | v1, 2018-08-14 |
| [sentencesNR.mat](https://osf.io/download/dgxb8/) | `/task2 - NR/Preprocessed/`; 595 sentence strings | v1, 2018-08-22 |
| [firstLevel_NR.m](https://osf.io/download/gan9v/) | `/scripts/Matlab_scripts/`; sentence extraction code | v4, 2018-08-23 |
| Twelve `results*_NR.mat` exports | `/task2 - NR/Matlab files/`; sentenceData content/rawData | Ten v2 files dated 2018-06-25; ZPH v3 and ZJS v1 dated 2019-06-20 |

The current official hashes of all twelve exports match the pinned local
download plan. The earlier source audit verified their actual local bytes and
all 3,356 prepared observations against their source MAT content and positions;
this investigation reuses that preserved audit rather than rereading all large
waveforms. The fresh inventories establish that these are associated files in
the same corpus/task, **not a single simultaneous release of interchangeable
text files**. The annotation CSV predates the exported files; there is no
correction notice in the inspected file metadata establishing it as their
authoritative replacement.

The [NEMAR-distributed annotation TSV](https://github.com/nemarDatasets/nm000278/blob/main/stimuli/relations_labels_task2.tsv)
has exactly the same 300 parsed field dictionaries as the OSF annotation CSV.
Its downloaded bytes and Git blob metadata are preserved. This is corroboration
of distribution of the annotation representation, not independent evidence of
the text displayed during acquisition.

## Sentence index correspondence

The [extraction script](https://osf.io/download/gan9v/) loads `sentencesNR.mat`
and removes its first five practice entries (lines 44–45). It assigns sentence
content from that array and raw EEG from `rawSentEEG` (lines 609–614). Direct
comparison confirms:

- Export IDs **0–299** equal `sentencesNR.mat` entries **6–305**, using MATLAB's
  one-based indexing: **300/300 exact strings**, covering **3,356/3,356** prepared
  observations. The MAT contains another 290 trailing strings beyond this
  studied subset; they are not silently included as experimental trials.
- Annotation CSV data row `i+1` maps to export ID `i`. Presentation CSV data row
  `i+6` maps to that same sentence. These are **data rows excluding headers**.
- The identifier values are under opposite column names in the two CSVs:
  annotation `sentence_id` equals presentation `paragraph_id`, and annotation
  `paragraph_id` equals presentation `sentence_id`, for **300/300** rows. The
  annotation `sentence_id` alone has only 175 unique values; its two-column
  identifier has 300 unique pairs. Do not join these files by the identically
  named column alone or treat those values as zero-based export positions.
- All **300/300** control-question flags agree after the mapping. For example,
  export ID 0 corresponds to annotation `(sentence_id=5, paragraph_id=4)` and
  presentation `(sentence_id=4, paragraph_id=5)` at data row 6.

Thus the same-task/subset mapping is supported; a task mix-up, practice offset
or index shift does not explain these 37 differences. The code is supporting
evidence, not proof that this exact script revision generated every export:
its subject list omits ZJS, and current file dates span different versions.
Exact text equality is the stronger result here. Original acquisition display
logs/screenshots and a producer-verified per-export text version were not
available, so literal on-screen wording remains unresolved even though **row
mapping is resolved**.

The presentation CSV is not UTF-8. Its only non-ASCII bytes are two `D1` and
three `D5` bytes. Interpreting them as MacRoman gives coherent em dashes and
right apostrophes, also present in the UTF-8 annotation CSV. This encoding is
**inferred, not publisher-declared**. Under that explicit interpretation,
591/595 presentation strings equal the source MAT strings. The four differences
are presentation zero-based rows 2, 103, 105 and 253: one practice sentence and
three exported sentences (IDs 98, 100 and 248). The remaining 297 experimental
presentation strings equal the export, including every substantive wording
discrepancy below.

## Classification and affected observations

These are mutually exclusive, manually reviewed primary categories:

- **Formatting-only:** punctuation, spacing, date separators, or corroborated
  punctuation-decoding damage, with the same intended words. This does not
  assert identical embeddings or keys.
- **Canonical-key differences:** orthographic/diacritic or German transliteration
  variants of the same intended lexical item, beyond the first category.
  This is an editorial equivalence judgement, not a new normalization rule.
- **Substantive wording:** changed word, name, digit or unit content; a proposed
  alternate spelling is not silently treated as punctuation.
- **Unresolved mappings:** rows without a defensible corresponding sentence.
  There are none in this set. Uncertainty about which wording was displayed is
  a separate limitation and does not imply an unknown row mapping.

The repository key remains SHA-256 of NFKC/casefold/whitespace-collapsed text.
**All 37** discrepancies change that key, including all 21 formatting rows.
Both complete text representations have 300 unique candidate keys. The category
name “canonical-key differences” therefore does not mean that only its eight
rows affect keys. No accent stripping or new canonicalizer was installed.
The older audit's 26/11 character-filter heuristic is preserved; it is not this
manual four-category classification.

Counts below use the **actual saved seed-42, within-ZuCo1 CNN subject split** in
`results/cnn_candidate_comparison_20261030/batch_real/splits.json`, checked for
disjoint, complete coverage of the 3,349-observation duration-QC manifest.
They are not regenerated splits or counts for the transfer experiments.
Each cell is **unique sentence strings / affected EEG observations**.

| Category | Train (2,245 total) | Validation (512 total) | Test (592 total) | QC cohort (3,349 total) | Prepared (3,356 total) |
|---|---:|---:|---:|---:|---:|
| Formatting-only | 21 / 156 | 21 / 37 | 21 / 41 | 21 / 234 | 21 / 234 |
| Canonical-key differences | 8 / 57 | 8 / 14 | 8 / 16 | 8 / 87 | 8 / 88 |
| Substantive wording | 8 / 55 | 8 / 14 | 8 / 16 | 8 / 85 | 8 / 86 |
| Unresolved mappings | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| **Total** | **37 / 268** | **37 / 65** | **37 / 73** | **37 / 406** | **37 / 408** |

Train subjects: ZDM, ZDN, ZGW, ZJM, ZJN, ZKH, ZMG, ZPH. Validation: ZJS, ZKW.
Test: ZAB, ZKB. Sentence strings recur across these subject splits: the three
37s are the **same 37 sentences**, not 111 distinct sentences or independent
cohorts. Repeated reader observations are not additional independent texts.

The difference between 408 prepared and 406 retained affected observations is
exactly **ZMG/71** (canonical-key category, 0.03 s) and **ZJM/275** (substantive,
0.08 s), both already excluded by duration QC. The five other existing duration
exclusions are ZDM/168 (0.06 s), ZDM/173 (0.065 s), ZDM/195 (0.05 s), ZDN/278
(0.08 s), and ZJS/250 (0.08 s); none is among the 37. No new exclusion was made,
and no test observations were removed for this investigation. The 244 missing
source sentence slots were never prepared observations.

### All 37 reviewed sentences

The following IDs are zero-based **export positions**, not the repeated CSV
`sentence_id` field. Full before/after strings, all source identifiers, keys,
source-MAT indices, reasons and per-split observation counts are in the
[37-row ledger](../results/text_discrepancy_investigation_20261030/analysis_final/discrepancies.csv).
The [observation ledger](../results/text_discrepancy_investigation_20261030/analysis_final/affected_observations.json)
enumerates all 408 affected identities and retained/excluded split status.

| Category / edit | Export IDs |
|---|---|
| Formatting: space in `long-time` / `long -time` | 10 |
| Formatting: ASCII hyphen / en dash in dates | 23, 45, 51, 61, 91, 106, 122, 140, 150, 167, 180, 194, 235, 241, 252, 295, 297 |
| Formatting: replacement character / right apostrophe or em dash | 98, 100, 248 |
| Canonical-key: Jose / José | 26 |
| Canonical-key: Eidgenoessische / Eidgenössische | 69 |
| Canonical-key: Wuerttemberg / Württemberg, including Baden- prefix | 71, 73 |
| Canonical-key: Zoonomia / Zoönomia, also date punctuation | 114 |
| Canonical-key: Adams-Onis / Adams-Onís | 154 |
| Canonical-key: nee / née | 251 |
| Canonical-key: Creteil / Créteil | 255 |
| Substantive: nVera-Ellen / Vera-Ellen | 24 |
| Substantive: wrote / write | 35 |
| Substantive: Maria / Marić | 64, 65, 66 |
| Substantive: km? / km² | 203 |
| Substantive: extra `111` before Senator, also date punctuation | 227 |
| Substantive: Ruthy / Réthy, also Leopold / Léopold and date punctuation | 275 |

Left/right strings refer to export/annotation, respectively. The presentation
CSV agrees with the **export** on all eight substantive rows. For example,
converting `wrote` to `write` would replace the source export's verb with a
different form merely to follow the annotation CSV. This is not justified by
the available acquisition evidence. Plausible factual spelling or grammar is
not enough to establish what a participant saw.

## Dataset decision and limited versioned repair proposal

There is **no support for replacing all 37 targets with annotation CSV text**.
The existing targets preserve the producer's exported sentenceData content,
match the source sentence MAT, and mostly match the separate presentation CSV.
Keep these as the reproducible export-faithful dataset. Existing weak retrieval
results cannot be attributed to these differences, or dismissed as invalid,
without a controlled, separately reported sensitivity experiment.

A narrower correction has stronger evidence: the MAT/export lost punctuation
to Unicode replacement characters in three sentences, while two source text
representations support their intended punctuation. Propose, **without creating
or adopting it now**, dataset version **`zuco1_nr_textrepair_v1`**:

| Export ID | Proposed text-only edit | Train observations | Validation | Test |
|---|---|---:|---:|---:|
| 98 | `Paul` + U+FFFD + `s` → `Paul’s` | 7 | 2 | 2 |
| 100 | `Maryland` + U+FFFD + `s`, `state` + U+FFFD + `s` → right apostrophes | 7 | 2 | 2 |
| 248 | `1902` + U+FFFD + `19` → `1902—19` | 8 | 2 | 2 |
| **Total** | **3 unique sentences, 4 character occurrences** | **22** | **6** | **6** |

This is a proposed **text-catalog encoding restoration**, supported by the
inferred MacRoman presentation CSV and agreeing UTF-8 annotation. It does not
prove original on-screen bytes or fix any EEG timing. It affects 34 prepared
and retained observations, not 34 unique sentences. The practice-only damaged
sentence is outside this dataset and must not be introduced as another trial.
Leave the other 34 discrepant experimental targets untouched in this limited
version, including `km?`, the names, and the extra digits: the presentation CSV
does not corroborate their annotation alternatives.

If adopted later, the version must contain a new immutable manifest referencing
the same EEG arrays; a three-row old/new text and key changelog linked to these
source hashes; all 34 affected observation identities; a separately named,
pinned-model text cache; new manifest/cache/candidate-pool digests; and saved
split provenance. Preserve the existing manifest, cache, checkpoints and
reports in place. Never repair the existing cache under its old name. The
subject split may keep its exact observations; any sentence-based split or
cross-release shared-sentence calculation must be reconstructed under the new
keys and audited for changes. This proposal is constrained by source evidence,
not chosen from validation/test performance.

If that version is adopted, new results would require:

1. Refitting and reselecting every **text-supervised ZuCo1 model** whose labels,
   embeddings or candidate pool depend on these keys: statistical/frozen-feature
   ridge, projection/alignment CNN, and other supervised baselines. Repeat
   real, Gaussian and shuffled conditions with matched settings and
   validation-only selection. For the recent CNN comparison, rerun all six
   loss/control fits, not just rescore the old real checkpoint.
2. Rebuilding transfer plan/pool provenance and rerunning **both directions** of
   original-grid and wider-grid transfer, including all 18 wider-grid
   evaluations, if making claims about the repaired dataset. ZuCo1 supplies
   supervised source targets in one direction and target queries/candidates
   in the other. A target-only edit may permit reuse of an unchanged ZuCo2
   source fit only after source split, normalization, selection and feature
   dependencies are verified unchanged; evaluation and pool metadata still
   need new outputs. Never carry over selected target-dependent results.
3. Recomputing relevant retrieval/control comparisons and downstream tables,
   figures and presentations from the new artifacts. Label them as a separate
   dataset version and a post-investigation sensitivity analysis; the existing
   test set has already been inspected. A higher single test score would not
   establish that either text representation is correct or that EEG succeeds.

Unchanged label-free EEG preprocessing and signal-only cached features do not
intrinsically need recomputation. They can be reused only with verified signal,
channel, normalization and split dependencies and explicit new-version
provenance; a cache compatibility check must not be bypassed by editing old
metadata. Unaffected ZuCo2-only experiments need no rerun. **No rerun is needed
merely to retain and report the existing export-faithful results. None was
launched here.** This audit does not resolve participant independence across
releases or establish reliable retrieval beyond the existing controls.

## Reproduction and validation

Using the existing `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_text_discrepancy_audit.py -q
.\.venv\Scripts\python.exe scripts/investigate_text_discrepancies.py --output results/text_discrepancy_investigation_20261030/analysis_final
```

Measured audit validation: **5 passed in 0.83 s**, no skips; these tests contain
no model fits. They check classification coverage, substantive edits, agreement
with the repository key rule, distinct-sentence versus observation counts, and
refusal to overwrite an output directory. This is a targeted audit check, not
a rerun of the previously recorded model/software suite. Reproduction must use
a fresh output path: the script refuses an existing directory.

The final audit also checks exact source offsets, unique composite identifiers,
the saved split's complete/disjoint cohort coverage, official source checksums,
and all twelve export hashes against the pinned plan. Its
[`preservation.json`](../results/text_discrepancy_investigation_20261030/analysis_final/preservation.json)
records SHA-256 values for **415 unchanged protected files**, including all
previous result files/checkpoints and the prepared/cohort manifests and text
cache. [`summary.json`](../results/text_discrepancy_investigation_20261030/analysis_final/summary.json)
pins script/test/input hashes and every count. `analysis/` is the preserved
initial counting pass; `analysis_final/` adds code/version provenance and
current-export checks with the same counts. New small source downloads and
original audit outputs are in this investigation folder. No previous experiment
file, target, split, checkpoint or result was changed.

The new [research archive](research_results/text_discrepancies_20261030/README.md)
also preserves 19 byte-verified copies of the final summaries, complete ledgers,
small source files/inventories, audit script and tests. Its separate
`validation.json` records the final input-hash, split-count, local-link and
preservation checks; `archive_manifest.json` records each source/copy hash.
