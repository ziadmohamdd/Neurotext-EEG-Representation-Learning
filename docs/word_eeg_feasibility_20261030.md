# Word/fixation EEG feasibility audit — 2026-10-30

**The local ZuCo sentence MAT files contain real fixation EEG cells and documented
word-level spectral features. A separately versioned, gaze-assisted view can
support sentence retrieval, subject to the coverage, boundary and leakage
conditions below. This is not yet an EEG-only word-segmentation pipeline or a
strictly window-matched replacement for the existing sentence CNN input.**

No model was trained or scored. No existing dataset, text target, split,
checkpoint or result was changed. The proposed punctuation repair remains
unapplied. Counts describe a read-only source census, not retrieval performance.

## Scope and measured coverage

The full census is complete. All 30 source MAT hashes match the pinned local download plans. Here, “usable” means a fully finite `(105,N)` fixation with `N≥2`, or eight fully finite 105-channel vectors for a selected reading-measure family. It does not establish physiological quality, semantic information or valid all-run timing.

| Coverage | ZuCo1 | ZuCo2 |
|---|---:|---:|
| Source sentence slots | 3,600 | 6,282 |
| Slots with word records | 3,500 | 6,083 |
| Word observations, repeated across readers | 76,891 | 119,271 |
| Slots with ≥1 usable raw fixation / TRT word | 3,487 | 6,019 |
| Existing prepared sentence observations | 3,356 | 5,911 |
| Existing duration-QC observations | 3,349 | 5,884 |
| QC observations with ≥1 usable raw fixation / TRT word | 3,342 | 5,850 |

There are **100 ZuCo1 and 199 ZuCo2 slots without word records**. Among all slots that do have word records, joining the words reproduces the parent sentence text exactly in **3,500/3,500** ZuCo1 and **6,083/6,083** ZuCo2 observations; their word-bound row counts also match. Thus the 299 absent-word slots are missingness, not observed contradictory text mappings. All existing prepared/QC targets match their source sentence text. Word observations repeat sentences and lexical items across readers; they are not unique word types or independent cohorts.

### Word-feature missingness across all exported word records

Each cell is **finite 8×105 / empty / all-NaN vector / absent field**. The same state counts were verified for all eight bands within each family; finite counts require all eight bands, not just theta. Denominators are 76,891 ZuCo1 and 119,271 ZuCo2 word observations.

| Family | ZuCo1 | ZuCo2 |
|---|---:|---:|
| FFD | 51,024 / 25,715 / 152 / 0 | 79,509 / 39,450 / 312 / 0 |
| SFD | 28,918 / 47,295 / 78 / 600 | 51,654 / 65,850 / 158 / 1,609 |
| GD | 51,048 / 25,715 / 128 / 0 | 79,535 / 39,450 / 286 / 0 |
| GPT | 51,073 / 25,715 / 103 / 0 | 79,595 / 39,450 / 226 / 0 |
| TRT | 51,089 / 25,715 / 87 / 0 | 79,639 / 39,450 / 182 / 0 |

Raw fixation cells contain **89,643 finite matrices and 306 scalar-NaN cells** in ZuCo1, and **122,213 finite matrices and 597 scalar-NaN cells** in ZuCo2. Finite snippet lengths range from **51–1,018 samples** and **51–1,293 samples**, respectively. No partly nonfinite raw matrix or partly nonfinite band vector was observed in this census. `rawEEG` itself is absent from **205 / 1,351** word records, rather than merely empty. Valid ZuCo1 raw cells are float32; ZuCo2 raw cells are float64. The spectral vectors are float64 when present. Sampled `rawET` cells are `(4,N)` in both releases; auxiliary row identities and a separate rawET cell-value quality census are not established here.

### Fixed ZuCo1 seed-42 retrieval split

| Split | Existing observations | With usable word EEG | Without usable word EEG | Complete TRT word observations |
|---|---:|---:|---:|---:|
| train | 2,245 | 2,244 | 1 | 32,848 |
| val | 512 | 512 | 0 | 7,101 |
| test | 592 | 586 | 6 | 8,734 |

All seven are word records with no fixated words and omission rate 1, not absent sentence targets: training **ZMG/70**; test **ZKB/71, 72, 81, 85, 93, 94** (zero-based sentence IDs). No new exclusions were made. In ZuCo2, 34 QC observations have no usable word EEG; their exact identities are in `derived_details.json`. The QC denominators remain 3,349 and 5,884.

Word features also exist for **144 ZuCo1 and 168 ZuCo2 observations outside the existing prepared sentence-EEG cohorts**. They may support a separately reviewed word-view cohort, but cannot be silently added to an architecture comparison. They remain outside the current datasets.



## Available fields and actual shapes

All shapes below use **MATLAB logical axis order**. HDF5 datasets expose reversed
physical axes: a stored `(samples, 105)` dataset is logically `(105, samples)`.
Treating that HDF5 shape as channels-first would corrupt the representation.
The audit covers every local NR export: twelve Level-5 MAT files for ZuCo1 and
eighteen MATLAB v7.3/HDF5 files for ZuCo2. It does not cover ZuCo1 sentiment
reading or either release's task-specific reading exports.

| Location | Fields / meaning | Shape or structure |
|---|---|---|
| `sentenceData` | `content`, `rawData`, `word`, `omissionRate`, `allFixations`, `wordbounds` | One struct per source sentence slot; unavailable slots retain text but may lack EEG/word data |
| Sentence spectral fields | `mean_{t1,t2,a1,a2,b1,b2,g1,g2}` and corresponding `_diff` | Eight `(1,105)` channel vectors and eight `(1,48)` difference vectors; inspect empty/NaN states rather than trusting field presence |
| `word` | Ordered word structs, including `content` | Normally `(1,W)`; word text is audit metadata, never a decoder input |
| `word.rawEEG` | One voltage matrix per fixation assigned to this word | Cell array `(1,F)`; valid cell `(105,N)`; invalid fixation may be scalar NaN; skipped words may have empty fields |
| `word.rawET` | Associated auxiliary/eye-tracking traces | Fixation cell array; sampled matrices have auxiliary-channel × sample axes; these are not EEG input channels |
| Word spectral fields | `{FFD,SFD,GD,GPT,TRT}_{t1,t2,a1,a2,b1,b2,g1,g2}` | Forty possible `(1,105)` vectors per word; one chosen reading-measure family yields **8×105 = 840** values |
| Word spectral differences | The same forty names with `_diff` | `(1,48)`, left-minus-right homologous electrode-pair features; not 48 additional recorded EEG channels |
| Word eye-tracking measures | `FFD`, `SFD`, `GD`, `GPT`, `TRT`, each `_pupilsize`; `meanPupilSize`, `nFixations`, `fixPositions` | Scalar durations/counts/pupil measures or fixation-index vectors, with empty/missing cases |
| `allFixations` | `x`, `y`, `duration`, `pupilsize` | Scalar struct containing vectors over recorded fixations; exported fields do not include the EEG start/stop arrays used by the producer |
| `wordbounds` | Displayed word bounding boxes | `(W,4)` when available; privileged stimulus-layout information |
| `omissionRate` | Fraction of displayed words not fixated | Scalar, derived from gaze and the text template |

The full field/shape/dtype and missing-state census is archived separately;
`[]`, a NaN sentinel and a NaN channel vector are distinct states. SFD is defined
only for a single fixation, so low SFD availability is not automatically a file
error. GPT can include EEG during regressions to earlier words; it is not
necessarily a measurement solely on the current word. “Raw” here means the
stored voltage snippets after the source processing, not verified untouched
acquisition data. No inverse transform turns these spectral vectors into raw
waveforms, phase, or millisecond-resolved EEG.
The existing waveform adapter deliberately reads `rawData`/`content`; a new
feature-view adapter is needed. A 105×8 band matrix must not be passed to the
waveform CNN as though its eight frequency bands were eight time samples.

There are **22 sentence-field names and 96 word-field names** across each
release, but a field need not exist in every sentence's word struct. Duration
units also need care: the ZuCo1 code explicitly applies a **50-sample (=100 ms)**
fixation threshold and slices both endpoints inclusively. Its stored FFD/TRT
values are sample counts, not milliseconds. In the recorded inspection examples,
FFD equals the first raw snippet's `N−1` and TRT equals the sum of `N−1` over
its cells in **35/35 ZuCo1 and 55/55 ZuCo2 applicable examples**. This supports
the same convention in the sampled ZuCo2 records, without certifying every
duration field's producer implementation. A duration-only control must not
silently interpret these numbers as milliseconds or use them as onset times.

## Channels, bands and feature definitions

All thirty existing subject schemas specify the same ordered **105 retained
scalp channels**, at a source sampling rate of **500 Hz**, with schema units
`uV`. The complete ordered labels and schema checksums are archived in
`channel_schemas.json`; they include EGI labels and Cz, not a guessed 105-channel
slice of an arbitrary montage. Sampled word/sentence waveform equality supports
consistent axes. The MAT word vectors do not themselves contain a separate
label for every channel, so this is corroborated source/schema correspondence,
not independent calibration or proof of every run's final reference state.

The [official ZuCo1 extraction code](https://osf.io/download/gan9v/) sets
`nChans=105`, takes `FullEEG.data(1:nChans, start:stop)`, filters continuous task
signals into the named bands, and uses mean absolute Hilbert analytic amplitude
over the chosen fixation. TRT then uses `nanmean` over fixation-level vectors:
**an unweighted mean of available fixation means**, not necessarily an average
weighted by each fixation's duration. The familiar “band power” label should
not be taken to mean squared amplitude or a new FFT estimate.
Recomputing a low-frequency spectrum on each short cropped snippet would not
reproduce filtering the continuous task before fixation averaging.

| Field suffix | Broad family | ZuCo1 script's filter limits, Hz |
|---|---|---:|
| `t1` | theta | 4–6 |
| `t2` | theta | 6.5–8 |
| `a1` | alpha | 8.5–10 |
| `a2` | alpha | 10.5–13 |
| `b1` | beta | 13.5–18 |
| `b2` | beta | 18.5–30 |
| `g1` | gamma | 30.5–40 |
| `g2` | gamma | **40.5–49.5 in code** |

The [ZuCo1 descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289117/)
documents fixation-linked Hilbert features and instead lists gamma2 as
**40–49.5 Hz**. Preserve this difference; do not silently round it away or
claim to have reproduced the numerical source transform. The inspected script
postdates several exports and omits ZJS from its subject list, so its exact
producer identity for every file remains unverified. All frequency limits above
are documented code definitions, not inferred from a 105-value vector.

The [official electrode-pair function](https://osf.io/download/6mpxe/)
defines all 48 ordered pairs, starting with E22−E9, E26−E2 and E23−E3.
Every listed electrode occurs in the retained channel schema. The source code
subtracts right from left; these signed differences must not be mistaken for
positive amplitudes or concatenated as anonymous extra electrodes. The pair
function and its hash, plus a machine-readable ordered mapping, are archived.

ZuCo2 has the same field names, vector dimensions and channel schemas. Its
[descriptor](https://aclanthology.org/2020.lrec-1.18/) supports the NR task and
simultaneous EEG/eye-tracking context. The inspected official OSF
`scripts/preprocessing` inventory was empty. The available official Python
readers demonstrate these field names, but do not establish the exact filter
implementation that produced every local ZuCo2 vector. Thus matching names is
enough to design an adapter, **not enough to certify identical spectral
processing between releases**.

The checksum-verified official Python helper also concatenates `_diff` fields
and four reading measures, defaults to float16, exposes literal word content
and word indices, and filters tokens with an alphanumeric regex. Its default
band-family concatenation is not the simple 840-dimensional single-measure
view proposed here. It should be evidence and a format example, not executed
unchanged as an EEG-only adapter. Never fall back to word embeddings or text
when EEG is absent, even if a downstream reader permits such records.

## Sentence-word and fixation mapping evidence

The producer assigns `word(ii).content` from the sentence's space-separated
tokens and uses displayed bounding boxes to assign gaze fixations to word
indices. Original sentence slot indices are retained in this audit and compared
with the existing prepared manifest texts; no CSV replacement, punctuation
repair, regex filtering or token reindexing is applied.

Do not confuse **lexical word order** with **chronological fixation order**.
Multiple fixations on one word are grouped together in that word's cell array;
reading regressions interleave them with other words. In the sampled ZAB
sentence, word 1/cell 2 begins at stored sentence sample 395, whereas word
3/cell 1 begins earlier, at sample 287. Flattening word-major cells would
therefore reorder the EEG timeline. `fixPositions` indexes the producer's
filtered word-fixation list, not an absolute EEG-clock onset; it is not safe to
join it directly to the unfiltered `allFixations` vectors as though their
indices had identical meanings.

The bounded corroboration checked five fixation cells from sentence 0 in ZAB
and five from sentence 0 in YAC, across all 105 channels. **Nine of ten had a
unique exact full-waveform match within the stored parent sentence.** The
remaining YAC word 2/cell 2 starts at parent-relative sample 4,631: its first
129 samples match exactly, while its 168-sample waveform extends **39 samples
past the 4,760-sample parent array**. This is measured overlap, not an invented
timestamp or a claim that the tail belongs to the next stimulus.

The original source code extracts fixation start-to-stop slices without
explicitly intersecting them with sentence end. Consequently, neither raw
fixation cells nor their supplied spectral summaries are automatically proven
to be sentence-contained. Fixation durations, positions and array lengths do
not establish the absent original recording start/stop indices. No −100 ms
baseline, inter-fixation gap, inter-trial interval or complete recording clock
was synthesized. All-run boundary validity remains unresolved; the ten-cell
probe must not be generalized into an all-corpus timing verification.

## Can this support the same sentence-retrieval task?

**Yes, conditionally, as a new measurement view with the same sentence labels
and retrieval candidates.** The smallest implementation would choose one
documented family, initially TRT, arrange its bands/channels explicitly, and
pool only finite 840-dimensional word vectors into one sentence representation.
A fixed-size pooled vector reduces direct exposure of the text-template length
compared with padding a full lexical word sequence, but does not eliminate
length, mask, duration or gaze-selection information.

For a comparison with the existing within-ZuCo1 experiment, retain its exact
3,349 observation IDs, seed-42 subject splits, unmodified sentence cache and
298 validation candidates. Do not quietly add source slots with word EEG but
no accepted sentence EEG, or drop hard validation/test queries. If a retained
observation lacks usable word data, a predeclared missing-input policy must
retain its query and candidate accounting. A broader recovered-word cohort
would be a separate protocol and dataset version, not an improved score on
the original cohort.

The task labels can stay the same while **the input window changes**: existing
CNN inputs use a center crop of at most eight seconds, whereas the supplied
word summaries use the producer's fixation windows across the reading period,
potentially including boundary tails. A result on supplied word spectra would
be a representation/window comparison, not evidence that alignment alone
improved the unchanged raw-CNN recipe. Precomputed spectra cannot be clipped
retroactively to an eight-second crop. Strict window matching would need
verified raw-fixation offsets and newly specified spectral extraction.

Raw fixation cells can support a later fixation/set encoder without inventing
gaps: encode each real snippet separately, then aggregate. For a strictly
sentence-contained version, verify unique waveform offsets or original event
latencies and intersect with the permitted sentence/crop window. Unmatched or
ambiguous cells require an explicit missing/quality state; do not assign a
plausible onset. Existing local continuous files cover only selected runs, so
an exhaustive recording-clock reconstruction may require more source data.

This remains **gaze-assisted, annotation-segmented EEG** even when only voltage
or spectral values enter the decoder. It does not establish word boundaries
from EEG alone, online/causal decoding, unseen-sentence generalization, or
cross-release participant independence. No retrieval benefit is established by
this availability audit.

## Training-only preprocessing contract

1. Save a separate immutable view/manifest with original observation IDs,
   source MAT hashes, word/fixation indices, band and channel order, field
   definitions, validity flags and the exact aggregation rule. Keep target
   text and all eye-tracking/text metadata outside the model-input whitelist.
2. Split at the original participant/sentence-observation level **before** any
   learned imputation, scaling, PCA, feature selection or outlier threshold.
   Never split a sentence's words/fixations across train/validation/test or
   count words as independent sentence examples. Existing held-out-reader
   splits repeat sentence classes; label that explicitly.
3. Fit imputation and an 840-feature scaler on training observations only;
   freeze them for validation/test and controls. Keep empty words, all-NaN
   vectors, partial nonfiniteness and absent word structs distinct in the
   provenance. If using log transforms, first verify the training values and
   signed-feature semantics; do not apply a positivity assumption to `_diff`.
4. Choose one family, aggregation, missing-input policy and any sequence cap
   using training/validation only. Do not use the test maximum word count,
   test missingness, test scores or pooled participant normalization to choose
   a model setting. Candidate construction and scoring remain sentence-level,
   with validation-only selection and frozen test handling.
5. For raw snippets, use the actual 500 Hz source rate, verified channel axes
   and a separate train-fitted voltage normalizer. Any deterministic resampling
   must preserve real episode boundaries. Do not concatenate across unknown
   gaps and filter the result as though it were continuous EEG.

“Training-only” here describes **new downstream transforms**. The supplied data
already underwent source recording-level processing, including synchronization,
filtering and denoising. An adapter cannot retroactively make the producer's
session-level preprocessing train-fitted. Continuous band filtering before
segmentation can also mix information across adjacent trial boundaries; that
matters particularly for future held-out-sentence/temporal claims. Keep this
upstream limitation distinct from avoidable leakage through a new global
scaler or explicit target text.

## Leakage risks and required controls

| Potential shortcut | Why it matters | Required diagnostic/control in a future approved experiment |
|---|---|---|
| Literal `content`, token IDs or word embeddings | They identify the retrieval target directly | Strict numeric EEG whitelist; schema/forward-path tests; text-only oracle separately labelled, never an EEG condition |
| Full lexical word count, empty word slots and positional masks | An annotation-derived sentence fingerprint survives even with EEG zeroed | Constant/zero EEG with identical masks/shapes; count-only and mask-only baselines; report candidate-count collisions |
| Fixation counts, duration and sequence length | Reading behavior can identify text or reader without EEG content | Count/duration-only controls; Gaussian EEG matched to real shapes, masks and amplitude structure; compare within length/mask strata |
| `rawET`, gaze coordinates, word boxes, pupil and omission rate | They carry eye movement and stimulus-layout information | Ban them from EEG inputs; an explicitly labelled ET-only condition and EEG+ET ablation if later requested |
| Fixation/word order and regressions | “Temporal” performance can come from known lexical alignment or gaze order | Within-sentence order shuffling and feature permutation preserving masks/counts; label lexical versus chronological representations |
| Incorrectly paired labels | Can reveal target/feature shortcuts and optimization artifacts | Consistent training sentence-class derangement, with unchanged correct validation targets and candidate pools, using matched initialization/budget |
| Missingness/artifact patterns | Dropped words/queries or imputed positions can encode sentence identity | Preserve query/candidate accounting; report all-empty cases; structural controls keep the same missingness masks |
| Subject/session/run identity or repeated sentences | Random word-level splits and within-reader leakage inflate apparent generalization | Exact original group splits, no identity metadata inputs; distinguish seen-class reader transfer from new sentence classes |
| Fixation tails and global preprocessing | A window may extend beyond its labelled sentence | Boundary audit and fixed clipping policy where offsets are verified; no invented timings or claims of exact window matching |

Gaussian and shuffled conditions alone are insufficient if they inadvertently
remove structural information preserved in real inputs. Controls must retain
the same number of episodes, allowed lexical slots, missingness, padding and
lengths; otherwise a “real-minus-noise” gap may just measure that mismatch.
Conversely, amplitude-matched Gaussian controls can retain nonsemantic
information, so they should accompany—not replace—the zero/mask/count controls.
Judge any later gain using validation results relative to all relevant controls
and participants, not training accuracy or a single higher test metric.

The static metadata audit makes the length shortcut concrete: the **298 validation candidates** have only **47 distinct word counts**, ranging from **6 to 65**. **Eight sentence classes are unique by word count alone**. If exact displayed word count is exposed, the largest remaining equal-count candidate bucket has only **16 sentences**. The training pool has 300 candidates with the same 47 counts and eight unique-count classes. These are collision counts, not fitted-model retrieval scores or an accuracy estimate.


## Implementation estimate

These are engineering estimates for one developer familiar with this repository,
not measured training runtimes or a promise of useful EEG performance.

| Work | Estimate | Acceptance condition |
|---|---:|---|
| Productionize bounded Level-5/HDF5 adapters and versioned feature contract | 1–2 days | Identical IDs/targets, explicit axes/bands, full missing-state handling and source hashes |
| Train-only scaling/imputation and fixed-size TRT sentence aggregation | 1–2 days | Frozen original splits and candidate pools; no text/ET in model tensors; no query deletion |
| Structural, Gaussian and class-shuffled controls plus leakage tests | 2–3 days | Matched masks/shapes and initialization/budget; metadata-only conditions clearly distinguished |
| Reproducible validation-only runner, provenance and report checks | 1 day | Frozen selection rule, per-reader results, test disabled until a justified later step |
| **Minimal spectral-view implementation** | **5–8 engineering days** | Ready for a separately authorized experiment; this audit does not implement/train that experiment |
| Raw-fixation path, additionally | **4–7 days** | Boundary/offset audit, episode-safe resampling/masking, raw encoder and window-comparability checks |

Obtaining missing continuous runs or resolving the exact ZuCo2 producer/filter
revision is an external dependency with no reliable duration estimate. Strict
all-run timing certification is not included in the spectral adapter estimate.
The audit readers are inspection tools, not a production training-data writer.
No new training runtime or accuracy was measured or invented.

A float32 840-feature family for all 196,162 exported word slots has an uncompressed
array payload of about **0.61 GiB**, before masks, metadata and indexing.
Pooling the 9,233 existing QC observations to 840 values each would require about
**29.6 MiB**. The finite raw-fixation samples would require approximately
**9.77 GiB** if converted to float32. These are storage arithmetic,
not a new cache, measured model memory usage or training-time benchmarks.

## Reproduction, preservation and audit limits

The audit used the existing `.venv`. **5 focused tests passed in 0.65 s** for
bounded nested Level-5 structs/cells, HDF5 axis/empty conventions, missingness,
scalar-struct handling and exact-containment rejection. No training tests or
model fits ran. All **30 MAT source hashes** match the pinned download plans;
the actual source files were opened read-only and their sizes/mtimes remained
unchanged. All **494 protected prior files** (results/checkpoints, manifests and
text caches) retained their hashes. Existing dirty worktree changes were preserved.

Commands for the completed census/consolidation included:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_word_eeg_audit.py -q
.\.venv\Scripts\python.exe -u scripts/audit_word_eeg.py --dataset zuco1 --output results/word_eeg_feasibility_20261030/zuco1
.\.venv\Scripts\python.exe -u scripts/audit_word_eeg.py --dataset zuco2 --output results/word_eeg_feasibility_20261030/zuco2
.\.venv\Scripts\python.exe -u scripts/audit_word_eeg.py --dataset zuco2 --workers 4 --completed results/word_eeg_feasibility_20261030/zuco2 --output results/word_eeg_feasibility_20261030/zuco2_parallel
.\.venv\Scripts\python.exe scripts/probe_word_eeg_containment.py
.\.venv\Scripts\python.exe scripts/summarize_word_eeg_audit.py
```

The initial sequential ZuCo2 audit was stopped to partition the remaining reads
among four bounded worker processes. Its completed YAC report and log were
preserved and counted once; incomplete work on YAG was not treated as a result.
The Level-5 reader was not changed semantically when parallel orchestration was
added. Source sentence `rawData` matrix values were skipped in the full word
census; only the ten-cell containment probe read its two parent waveforms.
Word/fixation EEG finite values and all named word bands were inspected fully.
Shape/finiteness does not verify original gaze classification, artifact removal,
run-specific reference, calibration or every acquisition-clock mapping.

The [audit archive](research_results/word_eeg_feasibility_20261030/README.md)
contains the full field census, sentence coverage, exact all-empty identities,
source hashes, channel and pair definitions, waveform probe, source readers and
code/configuration provenance. Per-subject inspection reports remain under
`results/word_eeg_feasibility_20261030/`. Output guards require fresh paths; no
old experiment file was patched or retargeted.

