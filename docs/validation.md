# Implementation validation and research status

Validated in the attached Linux cloud machine using Python 3.11.17, PyTorch 2.7.1+cpu and the versions in `tested_environment.json` / `requirements-lock-linux-py311.txt`.

- **81 tests passed**, including two optional official-weight LaBraM integration tests; none skipped in the configured cloud setup run. Upstream timm emits two deprecation warnings.
- Dataset software checks: versioned manifests, canonical sentence identities, MATLAB v5 structs and v7.3 HDF5 references, actual MNE BrainVision parsing of a tiny synthetic binary, alignment bounds, channel order, finite values and invalid-input rejection.
- Signal/model checks: resampling/filtering behavior, training-only streaming normalization, variable-length batches, padding-independent embeddings, quality-aware forward passes, masked reconstruction gradients, duplicate-aware alignment losses and classical features.
- Protocol checks: participant/content disjointness, LOSO folds, both transfer directions on synthetic fixtures, identity-audit failures, Gaussian and shuffled-class controls, known retrieval rankings/chance and subject bootstrap behavior.
- Integration checks: ridge fit/selection/evaluation, SSL training followed by matching-split alignment, separate control training, cached-feature projection and ridge, corruption evaluation, exact CPU epoch-interruption recovery, immutable content fingerprints after checksum-verified file staging, experiment execution/status export and all seven notebook schemas/Python cells.
- Official LaBraM source and weights are hash-verified; every inference backbone parameter loads strictly. Synthetic forward inference and frozen extraction for 14 input conditions pass. This is **not a real-EEG compatibility or semantic-performance result**.
- The complete `scripts/setup_cloud.sh` was executed on the actual cloud machine and rechecked for repeatability. Package dependency checks pass. Device detection reports CPU. The default experiment plan contains 288 jobs, all marked not executed; real-data jobs were not launched.

Run the default tests:

```bash
python -m pytest -q
```

The two official-weight tests are optional and skipped unless `EEG_LABRAM_CACHE` points to a complete verified cache. To reproduce the full cloud check:

```bash
EEG_LABRAM_CACHE=/workspace/eeg-env/labram /workspace/eeg-env/py311/bin/python -m pytest -q
```

Synthetic fixtures and their temporary metrics exist only to test software behavior. Test results are not reported as research accuracy. Scientific results directories contain no generated performance claims.

## Not executed / external prerequisites

The user has reported successful Windows ingestion, 200 Hz resampling, validation and sentence embedding generation for 217 ZJS observations, plus CUDA detection on the RTX 3050. Full-cohort ingestion, remaining source annotation/participant/montage audits, real-data baseline training, ZuCo SSL, real EEG/text alignment, quality/robustness comparisons and cross-dataset experiments remain unexecuted. Hugging Face sentence-model extraction awaits permitted model access or a verified local model directory. OSF, NEMAR website and Hugging Face connectivity attempts returned proxy tunnel 403; public GitHub metadata and LaBraM assets were accessible.

Windows installation and CUDA detection passed in user-reported output. RTX 3050 training, Colab GPU execution, peak-VRAM benchmarks and the full research matrix have not run. Notebook schema/cell checks do not imply a Colab training run. Environment draft persistence/publication and fresh-task restoration must be assessed separately from current-machine validation.

## Batch workflow verification

Batch tests cover v5/HDF5 EEGLAB metadata extraction, per-subject missing slots and audits, common-channel selection, complete/partial cohort outcomes, checksum-verified downloads, Range resume validation, disk preflight, shared text-pool assembly and reuse of completed subject caches. An end-to-end synthetic CLI test combines acquisition and preparation across three subjects; no synthetic result is presented as research performance. Live official download plans were inspected: ZuCo 1.0 NR has 12 subjects/24 selected files (~10.42 GiB); ZuCo 2.0 NR has 18 subjects/36 selected files (~34.43 GiB). Full cohort bytes were not downloaded in cloud validation. See `batch_workflow.md` before running on Windows.

## Incremental Level 5 fix

The user's full-cohort run confirmed that 10 of 12 sentence files exceeded the former v5 size guard; two prepared subjects contained 515 observations in total. The source sentence reader now streams selected fields rather than calling `loadmat` on the full struct. Fifteen new format/memory regression cases compare against SciPy, exercise compressed/uncompressed and endian/Unicode cases, check zlib corruption/truncation and field bounds, and verify a valid >512 MiB sparse-file fixture plus incremental skipped-field memory behavior. These are software fixtures, not scientific experiments. Existing prepared cache signatures remain compatible. OSF's cloud download redirect to Oracle object storage is currently blocked by the runtime proxy; the full real-cohort run must be retried on the already-downloaded Windows files, without another acquisition step.

## Ridge numerical stability

The first user-run real baseline completed with 592 held-out queries, but scikit-learn warned about an ill-conditioned single-precision solve. Its scores are retained as provisional: top-1 0.0067567568, top-5 0.0236486486, top-10 0.0371621622, MRR 0.0259074152. These user-reported numbers are not a statistical claim of above-chance decoding. Candidate-pool chance and subject/control analyses are required.

Ridge now casts only the compact regression features to float64 before standardization and uses SVD. Float32 disk signal/feature caches are retained. The source splits, normalization scope, candidate pools and validation-only alpha grid are unchanged; solver and precision are recorded in metadata. A regression fixture with duplicated/nearly collinear columns checks predictions against an independent double-precision SVD solution, and saved artifact coefficients are checked as float64. The real cohort must be refitted under a new output directory to confirm behavior on Windows. CLI evaluation now displays candidate count and analytic random-ranking chance alongside observed metrics.
