# Implementation validation and research status

Validated in the attached Linux cloud machine using Python 3.11.17, PyTorch 2.7.1+cpu and the versions in `tested_environment.json` / `requirements-lock-linux-py311.txt`.

- **48 tests passed**, including two optional official-weight LaBraM integration tests; none skipped in the configured cloud setup run. Upstream timm emits two deprecation warnings.
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

Real ZuCo ingestion, source annotation/participant/montage audits, real-data baseline training, ZuCo SSL, real EEG/text alignment, quality/robustness comparisons and cross-dataset experiments await actual data and verified sentence alignments. Hugging Face sentence-model extraction awaits permitted model access or a verified local model directory. OSF, NEMAR website and Hugging Face connectivity attempts returned proxy tunnel 403; public GitHub metadata and LaBraM assets were accessible.

Windows/RTX 3050 and Colab GPU runtime execution, peak-VRAM benchmarks and the full research matrix have not run. Notebook schema/cell checks do not imply a Colab training run. Environment draft persistence/publication and fresh-task restoration must be assessed separately from current-machine validation.
