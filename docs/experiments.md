# Execution and experiment reporting

## Staged execution

1. Install on CPU, run the unit/integration suite, and inspect device information.
2. Acquire one real subject/run per version, verify channels/reference/units and sentence-clock alignment; do not start full training before this audit passes.
3. Ingest and preprocess selected shards, inspect rejection reports and first-signal plots, verify subject/sentence reports, then extend to the permitted cohorts.
4. Save immutable cleaned manifests. Create separate within-version manifests, plus a combined manifest for both transfer directions. Restrict primary conditions to `nr`. Add audited participant IDs in provenance.
5. Precompute one text cache covering all selected unique sentences. Pin the actual text model revision and reuse the identical cache in all comparisons.
6. Run statistical/spectral ridge and compact CNN locally, then train SSL locally for a short real-data debugging run. Move larger schedules/repeated splits to Colab.
7. For LaBraM, audit electrode mapping and use the same source-channel subset for all representations. Its recommended filtering differs from the default; report this as a preprocessing factor, or use the same justified physical input recipe for all methods. Use one common final manifest when executing a controlled full matrix.
8. Save trained models, validation history, exact split and candidate-pool digests, device/dependency information and all controls. Evaluate selected checkpoints once on final clean and corrupted test sets. Inspect subject-level estimates before aggregating.

## Matrix

Default `experiments.yaml` defines 3 seeds x 4 representations x 4 protocols x 3 training controls x 2 quality settings = 288 jobs, with 12 robustness conditions per real/shuffled-control job (Gaussian training controls are evaluated on Gaussian inputs without this additional suite). This is an expensive research campaign, not a required local smoke check. Without `--execute`, the runner saves only `not_executed` specifications. With execution it records failures and stops without fabricating results. Use reduced YAML subsets while debugging.

The two within-dataset protocols run on whatever manifest you supply. To study both ZuCo versions separately, run each version's manifest independently with an experiment specification containing only those protocols. A combined-manifest held-out-subject run is a **pooled-dataset** study and must be labelled accordingly. Transfer jobs select source and target from the combined manifest. For a clean complete campaign, use separate within-version and transfer specs; see `configs/within_dataset.yaml` and `configs/transfer.yaml`.

`train-baseline --features` fits ridge to frozen features. `train-alignment --features` fits a contrastive projection to frozen features, sharing alignment implementation. Pretrained cache creation accepts `--include-controls --seed 42`; repeat with 43 and 44 so noise/masking agree across encoders. Pass quoted templates such as `--features "data/processed/labram_s{seed}" --quality-features "data/processed/labram_quality_s{seed}"` to the runner. Each cache must be made from the exact same complete manifest used for that run. The runner validates seed and quality setting rather than silently comparing differing corruptions.

Outputs include `config.json`, `splits.json`, `history.json`, `last.pt`, `best.pt`, ridge coefficients, evaluation JSON, per-subject metrics, cluster CIs and CSV exports. Robustness plots come from `evaluate --robustness`. Plot and discuss clean versus degraded curves for quality-aware/standard runs with matching candidate hashes; do not pool incompatible protocols. Exported files include data-origin status so synthetic validation cannot be mistaken for research data.

## Colab

Notebooks 04/05/06 clone your **published implementation commit**, install the same package, mount Drive, inspect GPU/free VRAM and accept persistent manifest/checkpoint paths. Configure `EEG_PROJECT_DRIVE` and paths in the cells. Raw data acquisition happens once in persistent storage. `stage_manifest` copies only referenced cached NPZ shards to `/content`, rewrites their references and verifies file checksums; it never fetches complete datasets on restart. Keep `num_workers=0` for initial debugging; benchmark two workers on Colab before increasing.

Save checkpoints directly to your Drive experiment directory. `--resume .../last.pt` resumes the original total epoch schedule at the next epoch, restoring optimizer/scaler/RNG. Drive state can outlive the runtime; GPU processes cannot. After an interruption, rerun setup cells, stage selected data, restore paths, inspect GPU availability, and resume. To move checkpoints between Windows and Colab, re-ingest manifests with the same stable observation identities and preserve the exact signal/text files. Split fingerprints bind observation metadata, preprocessing and actual cached bytes, while excluding machine-specific file locations. The tested staging utility rewrites paths without changing these fingerprints. Preserve immutable signals and the text cache; do not change hyperparameters or rewrite checkpoint digests to bypass resume checks.

## Results and discussion template

For each run record data versions/cohort sizes, rejected observations, subject and content overlap, candidate count/hash, channel map/reference/filtering, encoder provenance, seed and hardware. Report top-1/top-5/Recall@K/MRR, subject-wise values and justified CIs; compare real EEG against separately trained Gaussian/shuffled controls and analytic chance. Discuss degradation slopes, quality ablations, subject variability, transfer asymmetry and sentence overlap. Report unexecuted experiments explicitly and failed runs separately.

No results table or scientific discussion conclusion is filled in at implementation time. Dataset availability, not simulated accuracy, gates research execution.
