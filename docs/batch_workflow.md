# One-job preparation of normal-reading cohorts

The manual ZJS inspection established a real example: scalar-NaN sentence slots, 105 retained EEG electrodes, 500 Hz source data, and existing preprocessing. The batch workflow checks each subject instead of silently copying ZJS's metadata. It processes one source file at a time, with sentence arrays cached separately; “one job” does not mean loading the complete corpus into RAM.

## Windows commands

From VS Code PowerShell, update the editable checkout:

```powershell
git pull origin main
```

First verify batch preparation on the ZJS files you already downloaded:

```powershell
.\.venv\Scripts\python.exe -m eeg_decoding.cli prepare-batch --dataset zuco1 --root data/raw/zuco1 --units uV --already-preprocessed
```

This automatically discovers `resultsZJS_NR.mat` and its preprocessed EEG header recording, builds the schema, skips explicit missing EEG slots, preserves original sentence IDs, resamples with anti-aliasing to 200 Hz, validates the cache and prepares one deduplicated sentence-embedding cache. No additional bandpass/notch, clipping, normalization fitting, training or subject splitting happens during preparation. Original source files remain intact.

To see what the complete release needs, without downloading files:

```powershell
.\.venv\Scripts\python.exe -m eeg_decoding.cli download-zuco --dataset zuco1 --root data/raw/zuco1
```

The live ZuCo 1.0 NR listing inspected during implementation contains 12 subjects, 12 sentence exports and one selected metadata recording per subject: **24 files, approximately 10.42 GiB total source data**, including files already present. This is a source-size measurement, not an estimate of the processed cache size. Allow substantial additional disk space for caches and incomplete downloads; around 25 GiB free is a practical starting allowance, not a guaranteed maximum. Source listings can change, so inspect the freshly generated plan.

Then acquire and prepare the whole NR release in one command:

```powershell
.\.venv\Scripts\python.exe -m eeg_decoding.cli prepare-batch --dataset zuco1 --root data/raw/zuco1 --download --units uV --already-preprocessed
```

You can also download first with `download-zuco ... --execute`. Files already matching the OSF SHA256 are reused. `.partial` downloads resume when the server supports byte ranges; a server ignoring Range restarts the partial transfer. TLS verification and source-declared SHA256 remain enabled. Existing final files with a different checksum are preserved and reported, never overwritten. Correct the discrepancy or choose a clean release folder before retrying. Checksum-mismatched partial files are quarantined and require inspection/removal before another attempt.

If interrupted, rerun the same command. Verified downloads and completed subject caches are reused; an unfinished subject is reprocessed. Source bytes, metadata headers, target rate, electrode selection and pipeline version form the cache signature. Changed/corrupted prepared bytes are reported. The job requires Internet access to OSF and Hugging Face; it does not download raw continuous datasets or unrelated tasks.

## Outputs and model training

Default outputs live under `data/processed/zuco1_batch/`:

- `summary.json`: current cohort, each subject's outcome, imported/skipped counts, failures, selected channels, combined manifest and actual text-cache path.
- `combined.jsonl`: current complete local/requested cohort, written only when every discovered/requested subject succeeds.
- `partial.jsonl`: explicitly incomplete cohort when some subjects fail; no text cache is created for that incomplete run.
- `validation.json`: subject/sentence counts and signal-validation errors.
- One directory per subject: schema, inspected metadata/history, validation, restart state and processed sentence arrays. `state.json` contains skipped sentence IDs.
- `sentence_embeddings_<signature>.npz`: one shared deduplicated cache for the combined sentences. Different cohort/model settings receive a new immutable filename.

The summary is authoritative for the latest run. A previously successful `combined.jsonl` can remain on disk after a later partial run; do not treat an old file as completion of the new run. A nonzero exit and explicit failure summary distinguish partial acquisition/preparation from success. If only text-model download fails, prepared EEG remains usable and the next run reuses it while retrying text caching.

After successful preparation of at least three subjects:

```powershell
$eegBatch = Get-Content data\processed\zuco1_batch\summary.json -Raw | ConvertFrom-Json
.\.venv\Scripts\python.exe -m eeg_decoding.cli train-baseline --manifest $eegBatch.manifest --text-cache $eegBatch.text_cache --output checkpoints/zuco1_batch_ridge
```

Three subjects are only a minimum for separate train/validation/test groups, not a sufficiently powered research cohort. Training code creates deterministic subject-disjoint splits and fits normalization on training subjects only. Do not train from an incomplete report without explicitly defining a new cohort. Cached preparation does not mean experiments have executed.

## Metadata and scientific constraints

`--units uV` declares the verified convention of these ZuCo sentence exports; amplitude alone is never used to infer units. `--already-preprocessed` acknowledges that source conditioning is retained with no guessed original filter settings. The exported rawData field contains source-preprocessed waveforms, not untouched acquisition data. Batch reports retain inspected header histories, but minimal histories do not establish exact upstream filters or artifact-removal methods.

For each subject, at least one actual `*_SUBJECT_NR*_EEG.mat` recording is required to establish ordered labels and sampling frequency. The downloader selects the smallest available preprocessed EEG recording in that subject's folder. All matching headers you supply locally are checked for consistency. One supplied run is not an audit of absent runs: source release documentation and a full cross-run montage/reference audit remain external research requirements. No electrode order or Hz value is invented when headers are missing.

By default all prepared subjects must share the same electrode set. Order differences are corrected by verified electrode labels. Differing sets fail explicitly. Only use `--channel-policy intersection` when the scientific protocol explicitly chooses the common electrodes; that choice changes cache signatures and all models must use the resulting same manifest.

MATLAB v7.3 sentence references and EEG headers are lazy. Level 5 sentence exports are now streamed: only raw EEG and sentence text are materialized, while redundant word/band fields are discarded in 64 KiB chunks. Compressed-stream checksums, matrix sizes and selected field bounds are checked. The old 512 MiB subject-file restriction is replaced by a 64 MiB per-selected-field guard; unsupported layouts still fail explicitly rather than falling back to full-subject loading. The separate generic v5 EEGLAB metadata reader retains its 512 MiB guard; the downloader selects small metadata recordings. Nonfinite waveforms, inconsistent channels and empty subjects remain errors; scalar-NaN/empty EEG slots alone are skipped with an audit.

## ZuCo 2.0 and other conditions

The same command accepts `--dataset zuco2` with a separate input/output root and discovers the official normal-reading task rather than assuming task numbers match ZuCo 1.0. The OSF structures, filenames and required checks must still match; unsupported releases produce explicit errors. The default task is NR only; sentiment and task-specific reading are not treated as interchangeable conditions.

Cross-dataset participant-identity audits remain mandatory before transfer. Preparation does not invent `person_id` mappings. Real EEG compatibility, subject identity and downstream scientific performance remain to be validated on the actual downloaded cohort.

## Verification scope

Synthetic unit/integration tests exercise MATLAB v5/HDF5 metadata reading, missing slots, multi-subject preparation, shared text-pool assembly, montage handling, restart/cache verification, checksum-preserving resumable downloads and the combined download/preparation CLI. Live public OSF listings/checksum plans have been inspected. Full real-data cohort download/preparation and Windows execution of these new commands have not run in the cloud; the user-side ZJS batch run is the first intended real-data integration check. No scientific metrics are fabricated.

## Recovery from the former Level 5 size restriction

The first full Windows run downloaded the cohort successfully but prepared only ZDN (298 observations) and ZJS (217). The remaining subjects hit the former 512 MiB Level 5 guard. That was a loader limitation, not a reason to discard or redownload the data.

After updating with `git pull origin main`, rerun preparation without `--download`:

```powershell
.\.venv\Scripts\python.exe -m eeg_decoding.cli prepare-batch --dataset zuco1 --root data/raw/zuco1 --units uV --already-preprocessed
```

Completed caches retain their existing signatures and are reused. Failed subjects are processed with the bounded incremental reader. Progress is printed every 50 retained observations. Consume each source completely before trusting completion: the batch job only writes successful subject state after the entire iterator and compressed-stream integrity checks finish. A failed subject remains failed/partial even if some temporary sentence arrays were emitted before an error.

New regression coverage compares selected values with SciPy for compressed/uncompressed structs and cell-wrapped scalar structs, verifies little/big endian encodings and Unicode, rejects truncated/corrupt compression, retains per-field allocation guards, and exercises a valid file above the former 512 MiB threshold. A memory test traverses 48 MiB of irrelevant nested fields with less than 8 MiB peak traced allocations; this is reader-test allocation overhead, not a claim about total application RSS. Full real-cohort completion still needs confirmation on the user's downloaded files.
