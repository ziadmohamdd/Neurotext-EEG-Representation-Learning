# Implementation and verification plan

1. Establish Python packaging, local/Colab configuration and audit official source metadata.
2. Implement explicit MATLAB and BIDS adapters, bounded-memory preprocessing and versioned manifests.
3. Validate statistical/spectral ridge baseline and padding-safe compact CNN.
4. Validate masked temporal reconstruction with train-only SSL, early stopping and resumable checkpoints.
5. Verify official LaBraM weights/license, strict loading and frozen extraction; audit montage mapping before real EEG use.
6. Cache CPU sentence embeddings, train multi-positive semantic alignment, evaluate a fixed candidate pool.
7. Run amplitude-matched noise, shuffled-label and graded corruption controls using identical held-out observations.
8. Save subject/content split manifests; enforce zero-shot cross-dataset compatibility and participant identity audits.
9. Add shared-code notebooks for Windows and persistent Colab execution.
10. Run synthetic unit/integration tests and report their actual counts. Real-data experiments are blocked until ZuCo and verified alignment are available. Never substitute fixtures for scientific results.

Each stage is tested before expensive work. No data has been supplied; no research performance is claimed.
