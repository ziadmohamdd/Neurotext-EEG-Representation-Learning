# Block-holdout pilot v2: results

Run on 2026-11-05 with the frozen plan in `configs/block_holdout_pilot_v2_20261105.json` (SHA-256 `eef247790d718091b5f3e5357feb1989ceb7a3881f852868170ad89a641f9006`).

**Result: no model passed the gate on validation, so the test blocks were not scored.** With neighbouring sentences kept out of training, EEG did not beat the controls on these readers. All scores below are validation scores, reader-macro averaged, from familiar readers on new sentences.

## Scores

Expected N-way accuracy on 1,427 validation observations. CNN rows are the mean of three seeds. Other rows are one fit.

| Input | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| CNN, real EEG | 50.00 | 25.09 | 10.02 | 4.25 |
| CNN, matched noise | 49.80 | 24.82 | 9.72 | 3.94 |
| CNN, shuffled pairs | 49.74 | 24.80 | 9.98 | 4.18 |
| Ridge F, real EEG | 53.22 | 28.85 | 12.66 | 5.73 |
| Ridge F, matched noise | 53.62 | 28.75 | 12.41 | 5.46 |
| Ridge F, shuffled pairs | 50.06 | 25.28 | 10.34 | 4.18 |
| Ridge V, real EEG | 53.22 | 28.55 | 12.24 | 5.22 |
| Ridge V, matched noise | 52.74 | 27.84 | 11.45 | 4.82 |
| Ridge V, shuffled pairs | 50.51 | 25.76 | 10.76 | 4.61 |
| Word counts and timing, no EEG | 53.44 | 29.77 | 13.03 | 4.76 |
| List position, no EEG | 53.78 | 28.94 | 11.74 | 4.22 |

The highest 10-way score came from word counts and timing, which use no EEG. Ridge scored above chance on matched noise too, so its gain above chance does not come from the EEG waveform. The noise keeps each observation's channel amplitudes, which may explain this, but this run did not test that.

## Gate

The gate required all of these on validation, separately for each model. Every check failed for all three models.

| Check | Required | Ridge V | Ridge F | CNN F |
|---|---:|---:|---:|---:|
| 10-way gap over noise (points) | at least 2 | 0.79 | 0.25 | 0.31 |
| 10-way gap over shuffled pairs (points) | at least 2 | 1.48 | 2.32 | 0.04 |
| 10-way gap over timing (points) | at least 2 | -0.79 | -0.37 | -3.01 |
| 10-way gap over list position (points) | at least 2 | 0.51 | 0.93 | -1.71 |
| Within-block gap over noise (points) | at least 2 | 0.10 | 0.27 | 0.10 |
| Readers above noise | at least 20 of 30 | 17 | 15 | 18 |
| Reader bootstrap lower bound (points) | above 0 | -0.57 | -0.96 | -0.40 |
| Sentence bootstrap lower bound (points) | above 0 | -1.14 | -1.19 | -2.18 |

## Comparison with v1

| CNN on real EEG, validation | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| v1, random sentence holdout | 66.42 | 42.59 | 23.10 | 12.56 |
| v2, block holdout | 50.00 | 25.09 | 10.02 | 4.25 |

**This drop does not measure the reading-order leak alone.** v2 also trained on fewer sentences (307 against 432) and used different preprocessing. The two pilots also use different validation sentences. A further test is needed to separate these causes, and the claim that list order matches presentation order is still unverified.

The CNN's best validation loss came after the first epoch in all nine runs. Training loss kept falling while validation loss did not improve, so the model learned patterns that did not carry over to the held-out blocks.

The SemKey paper's scores (72.29 / 50.48 / 29.56 / 16.24) used a random text split, so they cannot be compared with v2. This run does not show how SemKey would score on a block split.

## Integrity

- The leakage audit passed with no failures: minimum list distance 11, maximum token Jaccard 0.41, maximum token containment 0.78, maximum text cosine 0.78.
- `decision.json` was written before any test scoring could run, and the test blocks were never scored, so they remain unused.
- All 11,054 protected files were unchanged (`preservation_after.json`).

## Files

- `validation_summary.csv`: every arm and seed
- `decision.json`: gate outcomes
- `leakage_audit.json`: split audit
- `plan.json`: frozen plan
- `v1_decision.json`: v1 gate outcomes for comparison
- `preservation_after.json`: protected-file check
