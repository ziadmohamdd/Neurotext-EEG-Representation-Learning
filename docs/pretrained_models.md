# Pretrained model audit

Selected model: official LaBraM-base, Jiang et al., ICLR 2024, https://github.com/935963004/LaBraM . Repository license at inspected revision is MIT; review model/data use terms for your institution. No third-party implementation is copied into this repository: verified official source files are downloaded to a cache.

Pinned revision: `c431221e6cfd23dbfa9950e0180682fb322b0548`.
Published checkpoint: `checkpoints/labram-base.pth` in that revision, accessible from the official GitHub repository. SHA256: `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.
Checkpoint is approximately 93 MiB including pretraining state; the extracted student inference backbone has 5,819,936 parameters (~22.2 MiB float32). Source and license checksums are also enforced before execution. PyTorch weights-only deserialization permits only the specific official Namespace/NumPy scalar metadata types; arbitrary pickle loading is not used.

Input contract from official code: microvolts divided by 100; 200 Hz; `[B,C,number_of_200_sample_patches,200]`; explicit standard-electrode positional indices. Official pretraining README prescribes 0.1–75 Hz, 50 Hz notch, 200 Hz. Remove unrelated channels, inspect reference scheme, and document montage mapping/domain shift. The student checkpoint contains `norm`, whereas the default fine-tuning architecture introduces `fc_norm`. Our extractor selects the pretrained norm configuration (`use_mean_pooling=False`), loads **every** backbone parameter strictly, and averages its normalized patch tokens. It does not leave a random or newly initialized backbone normalization layer under a pretrained label.

EGI E1–E128 labels are not the official standard-electrode vocabulary. Require an audited one-to-one source-electrode to standard-position mapping derived from actual electrode coordinates and reference documentation. Subset/harmonize **all** comparison models to the same mapped source channels if making a controlled comparison. The adapter rejects unrecognized/duplicate mappings. It cannot establish the scientific validity of a supplied mapping. No assumed EGI-to-10/20 mapping is shipped.

Before extraction, export the coordinates actually stored in all locally supplied
EEGLAB recording headers:

```powershell
python -m eeg_decoding.data.channel_audit --roots data/raw/zuco1 data/raw/zuco2 --output results/metrics/electrode_coordinates.json
python -c "from eeg_decoding.models.pretrained_eeg import acquire_official; print(acquire_official('checkpoints/official_labram'))"
```

The coordinate audit supports v5/v7.3 metadata, preserves channel order and source
hashes, and writes missing/empty/nonfinite coordinates as JSON null rather than
inventing positions. It does not load HDF5 EEG waveforms. For v5, the metadata
reader loads the selected EEG struct from the small recording file; its existing
512 MiB guard remains. Coordinate frame, origin, units, montage/reference and
cross-recording consistency must be verified before deriving a mapping. A full
XYZ count is not itself a scientifically verified mapping. The weight acquisition
command verifies the pinned official checkpoint/source checksums; it does not
train or extract features. Subsequent inference needs `pip install -e ".[labram]"`.

Frozen inference uses batch 1 and non-overlapping windows of up to eight seconds, drops any final fractional second, and duration-weights pooled window features. Reject inputs shorter than one second. At C=19 and eight seconds, attention has 153 tokens including CLS; at 128 channels it has 1025. A single float32 10-head attention matrix is roughly 0.9 MiB versus 40 MiB, respectively; intermediate allocations add memory. These are analytic lower-order estimates, **not GPU benchmarks**. CPU synthetic inference was validated; actual RTX 3050 and Colab peak memory must be measured. No pretrained fine-tuning is claimed; frozen features plus a small trainable projection are intentional for limited VRAM.

Real ZuCo/channel adaptation and semantic usefulness remain untested without data. Alternatives such as BENDR/EEGPT were not integrated because a single verified official-weight inference adapter suffices alongside the ZuCo-trained CNN/SSL comparisons; no compatibility or license claim is made for those alternatives.

## Uploaded ZuCo coordinates and explicit template adaptation

On 2026-10-28 the user supplied the actual coordinate audit containing all 30
locally selected recording headers (12 ZuCo1, 18 ZuCo2). All had identical ordered
105-channel XYZ arrays. After the EEGLAB axis conversion `[-Y, X, Z]` and fitting
one uniform scale, these matched MNE's `GSN-HydroCel-129` subset, with maximum
residual 6.45e-9 m. This identifies stored **template** coordinates; it does not
establish individually digitized locations, electrical reference, or complete
upstream filtering history. It does not resolve participant overlap.

`eeg_decoding.data.labram_inputs` implements a documented positional adaptation,
not an assertion of exact EGI-to-10/20 equivalence. Both MNE montages are transformed
using their native fiducials into head coordinates. Unit directions remove head-size
differences. A one-to-one minimum-angular-distance assignment selects 19 source
electrodes for conventional 10/20 tokens, with the named Cz vertex anchored to CZ.
The uploaded geometry gives directional errors of 2.11–9.40 degrees; the tool
rejects assignments above 12 degrees. This tolerance is an engineering choice,
not a clinical electrode-equivalence criterion. The method can differ from
published nominal EGI lookup tables. Use it as an exploratory foundation-model
adaptation and report the errors, not as an exact reproduction of a paper's montage.

The generated audit verifies **every** supplied header against the template with
a 0.1 mm residual limit and binds all input records to audited header hashes.
No arbitrary rotation or per-electrode deformation is fit. Each prepared signal
uses the selected real source channels, retains only complete seconds (matching
the adapter's temporal support), and excludes inputs under one second. All
exclusions and truncations are recorded. No new bandpass, notch, reference change
or train-fitted normalization is applied to these physical microvolt inputs.
This preserves upstream processing and must be reported as a potential mismatch
to LaBraM's published preprocessing recipe, whose reproduction is not claimed.

```powershell
python -m pip install -e ".[labram]"
python -m eeg_decoding.data.labram_inputs --manifest data/manifests/zuco2_duration_qc.jsonl --coordinate-audit results/metrics/electrode_coordinates.json --output data/processed/zuco2_labram19
python -m eeg_decoding.cli extract-pretrained --manifest data/processed/zuco2_labram19/smoke.jsonl --config configs/labram_full_sentence.yaml --channel-mapping data/processed/zuco2_labram19/mapping.json --output data/processed/zuco2_labram19_smoke_features
python -m eeg_decoding.cli extract-pretrained --manifest data/processed/zuco2_labram19/manifest.jsonl --config configs/labram_full_sentence.yaml --channel-mapping data/processed/zuco2_labram19/mapping.json --output data/processed/zuco2_labram19_features
```

Use new output directories on reruns; existing adaptation/feature artifacts are
protected. The smoke manifest includes the first retained signal and the longest
one. Whole-sentence extraction uses bounded windows and prints progress every 50
observations. Default extraction is real EEG only; `--include-controls` produces
14 variants and requires a separate output directory. No pretrained extraction
or decoding on user EEG has yet been executed in the cloud. Real-data completion
must be confirmed from the user's outputs.

For a fair representation comparison, fit a new statistical ridge baseline on
the **generated 19-channel manifest**, with `max_samples: null`, and compare it
with ridge/projection on its frozen LaBraM cache. Do not attribute differences
against earlier 105-channel models solely to pretraining. Preserve the same text
cache, split/seed and candidate pool. Repeat this adaptation for ZuCo1 before
claiming a comparable two-version experiment.

## Efficient matched negative controls

`extract-pretrained --include-gaussian` extracts only real/Gaussian variants,
without the 12 robustness conditions enabled by `--include-controls`.
`--reuse-real-features EXISTING_CACHE` copies a completed real feature array to
the new output after verifying model/checkpoint, manifest and observation order,
mapping, temporal crop, quality setting, seed and units. Shape, float32 dtype,
finite values and copied-file SHA256 are checked. Source caches are unchanged.
This avoids recomputing already completed real-data model inference.

For the user-reported ZuCo2 prepared directory, for example:

```powershell
$inputs = "data/processed/zuco2_labram19_20261029_015559"
python -m eeg_decoding.cli extract-pretrained --manifest "$inputs/manifest.jsonl" --config configs/labram_full_sentence.yaml --channel-mapping "$inputs/mapping.json" --reuse-real-features "${inputs}_features" --include-gaussian --output "${inputs}_control_features"
```

Fit a Gaussian ridge using `train-baseline --control gaussian --features NEW_CACHE`
and a shuffled-pair ridge using `--control shuffled --features NEW_CACHE`.
Evaluation defaults to Gaussian features for the Gaussian-trained artifact and
real features/true sentence labels for the shuffled-trained artifact. Keep the
identical prepared manifest, text cache and configuration. The real-condition
reference is the earlier matched LaBraM ridge, not the 105-channel baseline.
Reuse checks were tested with synthetic official-weight extraction; real-data
control completion still needs the user's actual execution output.
