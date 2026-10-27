# Data acquisition, inspected structures and unresolved alignment

## Official sources

- ZuCo 1.0: https://osf.io/q3zws/overview ; Scientific Data 2018, https://doi.org/10.1038/sdata.2018.291
- ZuCo 2.0: https://osf.io/2urht/overview ; LREC 2020, https://aclanthology.org/2020.lrec-1.18/
- NEMAR: https://www.nemar.org/dataset/nm000278 and https://www.nemar.org/dataset/nm000279
- Official NEMAR BIDS metadata: https://github.com/nemarDatasets/nm000278 and https://github.com/nemarDatasets/nm000279
- ZuCo authors' processing code: https://github.com/DS3Lab/zuco-nlp

Network inspection: OSF pages and NEMAR's website returned proxy tunnel 403; their full current documentation and OSF binary samples could not be inspected. GitHub NEMAR metadata was accessible and examined directly. Do not interpret this as a complete audit of the OSF MATLAB schema. Local MATLAB parser tests validate supported schema behavior, not a real OSF recording.

## Verified NEMAR structure

Inspection snapshots:

- nm000278 commit `fdeb3a97577fa4bc17fb2570b934ceb771d6d0b8`
- nm000279 commit `413a8952ea2393c403d4496ecf1c9ce8cd56aecf`

Files include `dataset_description.json`, `participants.tsv`, `stimuli/`, `derivatives/reading_measures/`, and `sub-*/eeg/*_task-*_run-*_eeg.{vhdr,vmrk,eeg}`, `_eeg.json`, `_channels.tsv`, `_events.tsv`.

README describes 128-channel EGI Geodesic EEG, 500 Hz, Cz reference and simultaneous eye tracking. The inspected nm000278 sidecar states `SamplingFrequency: 500`, `EEGChannelCount: 128`, but `Manufacturer: Brain Products`; preserve this metadata discrepancy and inspect the actual montage/header before choosing electrodes. Channel TSV names are E1, E2, … with microvolt units. MNE returns volts, which the loader converts to microvolts.

Task codes in ZuCo 1.0: `sr` sentiment reading, `nr` normal reading, `tsr` task-specific reading. ZuCo 2.0 normal and task-specific reading are candidate comparators; verify task definitions from sidecars and source documentation. Use matching `nr` conditions first, never pool `sr` into transfer as equivalent to `nr`.

The inspected `sub-ZAB_task-nr_run-1_events.tsv` columns are `onset,duration,trial_type,value,gaze_x,gaze_y,pupil_size,sacc_endx,sacc_endy,sacc_amplitude,sacc_peakvel`. Rows include `trigger_90`, `trigger_12`, fixations, saccades and blinks. **There is no sentence text column.** Derivative word reading measures are not automatically verified sentence-clock intervals. A safe loader cannot infer trigger-to-sentence timing and correspondence from this table alone.

Binary BrainVision files in these Git trees use git-annex references (e.g. `/annex/objects/SHA256E-...`); a metadata clone alone is not usable EEG. Use NEMAR's documented dataset download or configured git-annex retrieval, and confirm actual files resolve before ingestion. Do not treat symlink text as EEG. Download only selected subjects/runs to begin; full acquisition can be large. Review the CC-BY-4.0 NEMAR attribution and original dataset terms.

ZuCo 1.0 subject labels observed: ZAB, ZDM, ZDN, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW, ZMG, ZPH.
ZuCo 2.0 labels observed: YAC, YAG, YAK, YDG, YDR, YFR, YFS, YHS, YIS, YLS, YMD, YMS, YRH, YRK, YRP, YSD, YSL, YTL.
The strings do not overlap. **Actual participant overlap remains unaudited**; different IDs do not prove different humans. Cross-dataset code requires explicit verified `person_id` provenance and checks disjoint identity values.

## Required BIDS sentence mapping

Provide a UTF-8 TSV with columns:

```text
recording  onset  duration  sentence_id  text  subject  session  task
```

Use real tab delimiters. `recording` is a path relative to the dataset root, `onset/duration` are seconds in the EEG clock, and text/ID come from the verified stimulus alignment. Preserve run identity in a unique session or sentence observation ID and provenance. Verify multiple sampled boundaries against stimulus markers/source annotations. Do not invent durations, sentence text or channel labels. Optional `person_id` supplies an independently audited identity. Reader retains one open recording and retrieves only each interval, without preloading the corpus.

## MATLAB input contract

First inspect downloaded subject/task files with MATLAB, `scipy.io.whosmat` for v5 or `h5py` for v7.3. This adapter supports a `sentenceData` struct with explicitly named raw EEG and text fields, not arbitrary word-level bandpower exports. Provide schema JSON for each actual file:

```json
{
  "subject": "REPLACE_WITH_ACTUAL_ID",
  "session": "REPLACE_WITH_ACTUAL_SESSION",
  "task": "nr",
  "channels": ["REPLACE_WITH_ACTUAL_CHANNELS_IN_ORDER"],
  "sfreq": 500,
  "signal_field": "rawData",
  "text_field": "content",
  "time_axis": 1,
  "units": "uV"
}
```

These are **schema examples, not a verified description of every OSF export**. Determine actual field paths, retained channels, sample rates, units, preprocessing history and skipped sentences. If raw sentence signals are absent, use original continuous EEG plus verified events or export verified sentence shards. The loader fails rather than turning spectral bands into artificial waveforms. V7.3 references are read one sentence at a time; SciPy v5 loads one bounded subject/task variable, so files above 512 MiB are rejected. Compressed v5 can expand substantially; use v7.3 sentence shards on a 16 GB machine.

## Validation and versioning

Versioned JSONL records contain dataset, subject, session, task, sentence ID/text, absolute signal cache reference, ordered channels, sample rate, signal quality and provenance. Signals are float32 compressed NPZ per observation. Validation checks observation identity, dimensions, finite values, metadata, subject counts and sentence counts. Training saves content-and-signal digests and exact split membership; evaluation rejects changed test data. Manifests are local generated artifacts; preserve an immutable research copy with dataset licenses in your own storage. Never publish sensitive data without checking dataset terms.
