# NeuroText: EEG Representation Learning

Memory-conscious ZuCo 1.0/2.0 EEG-to-sentence retrieval for Windows (RTX 3050, 4 GB), CPU and Google Colab. One package implements all workflows; notebooks are clients of the same modules.

**Status:** user-side Windows experiments cover both ZuCo versions, statistical/CNN/SSL models, frozen official LaBraM features, matched negative controls for these configurations, and selected robustness comparisons. The [draft research report](docs/research_results/REPORT.md) archives 79 evaluation reports and documents near-chance retrieval and remaining work; these results were not independently rerun in the cloud. Strict cross-dataset transfer remains unexecuted pending participant-identity evidence. Automated software tests use explicitly synthetic fixtures. See [data acquisition](docs/data_sources.md), [methodology](docs/methodology.md), and [pretrained adaptation](docs/pretrained_models.md).

Repository: https://github.com/ziadmohamdd/Neurotext-EEG-Representation-Learning . Existing local folders can keep their current name. Fresh clones can preserve the documented checkout path with:

```powershell
git clone https://github.com/ziadmohamdd/Neurotext-EEG-Representation-Learning.git New_trends_AI_project1
```

## Windows setup (PowerShell, Python 3.11)

```powershell
cd New_trends_AI_project1
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CPU workflow:
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
# OR RTX 3050, with a compatible NVIDIA driver (see the official PyTorch selector):
# python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ".[dev,text,labram]"
python -m pytest -q
python -m eeg_decoding.cli device
```

For Linux/CPU replace activation with `source .venv/bin/activate`. This cloud instance also has a prepared Python 3.11 environment at `/workspace/eeg-env/py311`. Tested dependency versions are recorded in `docs/tested_environment.json`; GPU memory and Windows behavior must be benchmarked on the actual machine. The base configuration starts at batch 2, accumulation 8, embedding 128, workers 0 and 8-second crops. These are initial settings, not a measured 4 GB guarantee.

## Recommended batch workflow

Use one command to prepare all downloaded normal-reading subjects, with subject-specific metadata checks and bounded-memory caching:

```powershell
.\.venv\Scripts\python.exe -m eeg_decoding.cli prepare-batch --dataset zuco1 --root data/raw/zuco1 --units uV --already-preprocessed
```

Add `--download` to acquire missing official NR exports and one metadata recording per subject. First inspect the live size/checksum plan with `python -m eeg_decoding.cli download-zuco --dataset zuco1 --root data/raw/zuco1`. The inspected ZuCo 1.0 plan is approximately 10.42 GiB of source files, plus processed caches. No training happens during preparation. See [batch_workflow.md](docs/batch_workflow.md) for full-cohort commands, restart behavior, reports and limitations.

## Acquire and validate real EEG

Follow [data_sources.md](docs/data_sources.md) before ingestion. Raw BIDS and OSF MATLAB are different formats. **Do not use trigger-only events as sentence alignments.** NEMAR's inspected table has no sentence text, so supply a verified sentence-clock TSV. MATLAB requires an inspected schema JSON and raw sentence-level signals; band features are not waveforms. For exports containing scalar-NaN/empty EEG slots, add `--skip-missing-eeg` to ingestion; excluded sentence IDs are saved in the adjacent `.import.json` report.

```powershell
python scripts/inspect_data.py --root data/raw/zuco1 --output results/metrics/raw_inspection.json
python -m eeg_decoding.cli ingest --format bids --dataset zuco1 --root data/raw/zuco1 --alignment data/raw/zuco1_sentence_alignment.tsv --output data/manifests/zuco1.jsonl
# For one inspected OSF MATLAB subject/task file:
# python -m eeg_decoding.cli ingest --format mat --dataset zuco1 --root data/raw/resultsSUBJECT_NR.mat --schema data/raw/schema.json --output data/manifests/subject.jsonl
python scripts/inspect_data.py --manifest data/manifests/zuco1.jsonl
python scripts/preprocess.py --manifest data/manifests/zuco1.jsonl --channels data/raw/shared_channels.json --output data/manifests/zuco1_clean.jsonl
```

The channels file is a JSON list in a scientifically chosen shared electrode order. Restrict both versions to comparable normal-reading (`nr`) tasks for primary transfer. Merge cleaned JSONL manifests with `write_manifest` and `read_manifest` (absolute signal references are preserved). Do not combine all EEG into RAM. Add verified `person_id` metadata before transfer; different subject labels alone do not establish distinct people. No data or weights are committed.

## Train, resume and evaluate

```powershell
python -m eeg_decoding.cli cache-text --manifest data/manifests/combined_clean.jsonl --output data/processed/sentence_embeddings.npz
python scripts/train_baseline.py --manifest data/manifests/combined_clean.jsonl --text-cache data/processed/sentence_embeddings.npz --output checkpoints/ridge
python scripts/train_ssl.py --manifest data/manifests/combined_clean.jsonl --output checkpoints/ssl
python scripts/train_ssl.py --manifest data/manifests/combined_clean.jsonl --output checkpoints/ssl --resume checkpoints/ssl/last.pt
python scripts/train_alignment.py --manifest data/manifests/combined_clean.jsonl --text-cache data/processed/sentence_embeddings.npz --ssl-checkpoint checkpoints/ssl/best.pt --output checkpoints/ssl_aligned
# Omit --ssl-checkpoint for the supervised compact-CNN comparison.
python scripts/evaluate.py --manifest data/manifests/combined_clean.jsonl --artifact checkpoints/ssl_aligned/best.pt --text-cache data/processed/sentence_embeddings.npz --output results/metrics/ssl.json --robustness
python scripts/evaluate.py --manifest data/manifests/combined_clean.jsonl --artifact checkpoints/ssl_aligned/best.pt --text-cache data/processed/sentence_embeddings.npz --output results/metrics/ssl_noise_input.json --control gaussian
python -m eeg_decoding.cli export-results --root results --output results/tables/retrieval.csv
```

Use a **single-version manifest** for within-dataset experiments; the combined manifest is for an explicitly pooled-subject study or directional transfer. Use `--protocol subject_sentence` to hold out both participants and sentence content. Use `--protocol transfer --source zuco1 --target zuco2`, then reverse versions, for zero-shot transfer. Source validation subjects select models; target data never fits SSL, signal normalization or projection. Candidate pools are the unique sentences in the saved test split and are identical across model comparisons with identical splits. Retrieval is **not unrestricted text generation**.

`--control gaussian` on training commands trains a separate amplitude-matched noise-input model. `--control shuffled` trains deranged sentence-class pairs, while validation and test retain true pairs. Evaluation defaults to Gaussian input for Gaussian-trained checkpoints and real input otherwise; its shuffled option is an additional label-permutation diagnostic. `--quality-aware` enables label-free transient attenuation; compare against the same command without it. All train commands accept these controls; all corruption experiments use checkpointed models without fitting to degraded test data.

Checkpoints save optimizer, AMP scaler, RNG state, normalizer, splits, model settings and history. Resume `last.pt` with the original configuration, paths and total epoch schedule; epoch-boundary recovery is exact in the CPU integration test. Incompatible resumes fail explicitly. `best.pt` uses validation loss only. Each training run owns its output directory; do not share one between experiments.

## Frozen pretrained comparison

See [pretrained_models.md](docs/pretrained_models.md) before using LaBraM. EGI E1…E128 names are not its standard-electrode vocabulary. Supply an electrode-position-audited, one-to-one mapping and subset the manifest to those channels. **Do not rename arbitrary electrodes.** Use a separate LaBraM preprocessing config (example below), documented as a different input recipe.

```powershell
python -m eeg_decoding.cli extract-pretrained --manifest data/manifests/labram_clean.jsonl --config configs/labram.yaml --channel-mapping data/raw/audited_mapping.json --output data/processed/labram_features --include-controls
python scripts/train_alignment.py --manifest data/manifests/labram_clean.jsonl --text-cache data/processed/sentence_embeddings.npz --features data/processed/labram_features --output checkpoints/labram_alignment
python scripts/evaluate.py --manifest data/manifests/labram_clean.jsonl --artifact checkpoints/labram_alignment/best.pt --text-cache data/processed/sentence_embeddings.npz --features data/processed/labram_features --output results/metrics/labram.json --robustness
```

A train-only feature standardizer and a trainable contrastive projection operate on frozen cached features. Ridge also accepts `--features`. For a quality-aware pretrained cache set `quality_aware: true` during extraction and training, and use a separate cache directory. Re-extract controls for each experimental seed to align corruption draws across representations. Real EEG compatibility is not established by synthetic standard-channel tests.

## Colab and experiment matrix

Commit/push the implementation to your GitHub repository before cloning it from Colab. Open notebooks 04–06, mount Drive, set persistent manifest/text-cache/checkpoint paths and run the shared commands. No notebook duplicates model implementations. Stage only selected cached sentence shards locally; keep raw datasets and checkpoints in Drive. See [experiments.md](docs/experiments.md).

```powershell
# Save the full matrix as not_executed jobs (no invented results):
python scripts/run_experiments.py --output results/experiment_plan
# Execute only after data, identity audits, pretrained mappings and caches are ready:
python scripts/run_experiments.py --manifest data/manifests/combined_clean.jsonl --text-cache data/processed/sentence_embeddings.npz --features "data/processed/labram_s{seed}" --quality-features "data/processed/labram_quality_s{seed}" --execute
```

The full default matrix is expensive; edit the experiments YAML for a small debugging subset, then use Colab for the full study. The runner records failed/not-executed/executed states and never converts a failed command into success. Do not compare caches made from different manifests as if preprocessing were controlled.
