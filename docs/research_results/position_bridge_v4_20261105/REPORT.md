# Position bridge test v4: results

Run on 2026-11-05 with the frozen plan in `configs/position_bridge_v4_20261105.json` (SHA-256 `203d9d25d3a1ad75cf18fc2ecac50f0598c89d51ee8f233c39ab71fea5318d8e`).

**Result: position explains part of the CNN score.** A model that maps EEG only to list position, then copies the text of training sentences at that position, scored 15.41% at 10-way. Its shuffled-position control scored 9.70%, and the v1 CNN scored 21.10% on the same queries. Position alone did not reach the CNN.

**Main finding: EEG predicts where a held-out sentence sits in the list.** The correlation between predicted and true position was 0.80 in ZuCo 1.0 and 0.88 in ZuCo 2.0, for familiar readers.

These are validation scores on reused sentences, from readers who were in training. The bridge is not EEG decoding of meaning, and nothing here is a new accuracy claim. No test data was scored.

## Scores

Expected N-way accuracy, averaged within reader and then across 30 readers, on 766 observations of 53 sentences. Sentences in v1 test and the v2 test zone were removed first.

| Predictor | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| Saved v1 CNN, real EEG, re-predicted | 65.84 | 41.34 | 21.10 | 11.23 |
| Bridge, v1 input, real EEG | 55.48 | 31.47 | 15.41 | 7.77 |
| Bridge, v1 input, matched noise | 50.89 | 26.46 | 11.93 | 6.53 |
| Bridge, v1 input, shuffled positions | 49.80 | 24.77 | 9.70 | 3.94 |
| Bridge, F input, real EEG | 54.88 | 30.27 | 14.19 | 7.44 |
| Bridge, F input, matched noise | 51.86 | 26.94 | 12.22 | 6.55 |
| Bridge, F input, shuffled positions | 49.97 | 25.18 | 10.20 | 4.23 |
| Neighbouring text with true position, no EEG | 84.05 | 69.80 | 55.10 | 44.14 |

Scored against these kept sentences, the CNN's 10-way score is 21.10%, not the 23.10% it scored on all 81 validation sentences. The neighbouring-text score of 55.10% replaces the 53.94% in the v3 report, which included protected sentences.

## Decision

| Comparison at 10-way | Points | Reader 95% interval | Sentence 95% interval |
|---|---:|---:|---:|
| Bridge minus shuffled, v1 input (primary) | 5.72 | 3.29 to 8.23 | 1.50 to 9.74 |
| Bridge minus shuffled, F input | 3.99 | 1.97 to 6.02 | -0.10 to 8.10 |
| CNN minus bridge, v1 input | 5.69 | 3.12 to 8.09 | -1.31 to 13.24 |
| Bridge real minus noise, v1 input | 3.48 | 0.77 to 6.16 | 0.31 to 6.54 |
| Bridge real minus noise, F input | 1.97 | -0.15 to 4.29 | -1.48 to 5.29 |

The primary bridge beat shuffled by more than 2 points with both lower bounds above 0, but sat more than 2 points below the CNN, which gives the frozen label. The F input misses the same requirement, as its sentence bound just crosses 0.

Above chance, the bridge reaches 5.41 points against the CNN's 11.10, or about half. This does not mean half of the CNN's score came from position. It shows how far this one route can go.

## Position quality

| Arm, v1 input | ZuCo 1.0 error, sentences | ZuCo 1.0 correlation | ZuCo 2.0 error, sentences | ZuCo 2.0 correlation |
|---|---:|---:|---:|---:|
| Always guessing the middle | 39.3 | n/a | 87.9 | n/a |
| Real EEG | 21.3 | 0.80 | 36.9 | 0.88 |
| Matched noise | 37.3 | 0.54 | 61.7 | 0.69 |

- **Coarse but strong:** position is off by about 21 to 37 sentences. That is far too coarse to find the true neighbours, which is why the bridge stays well below the 55.10% that true position allows.
- **Noise reaches it too:** matched noise keeps each observation's channel amplitudes and still correlates at 0.54 to 0.69. So much of the position signal survives without the waveform. Real EEG adds to it.

## Other evidence

**Does the CNN follow the bridge?** Across the 53 sentences, the two scores correlate with a Spearman rho of 0.29 (p = 0.034, unadjusted). This is weak, and it was one of several secondary checks.

**How much does the split matter?** Neighbouring text over 20 fresh random splits, using no protected sentence and no EEG:

| Held out per split | 2-way | 4-way | 10-way mean | 10-way range |
|---|---:|---:|---:|---:|
| 10% | 83.00 | 68.68 | 51.09 | 38.36 to 65.62 |
| 15% | 82.64 | 67.13 | 48.79 | 37.37 to 62.00 |
| 25% | 78.92 | 62.08 | 43.73 | 36.07 to 51.62 |

So the high score is not a quirk of the v1 seed.

**Do sentence IDs follow the session?** IDs are each reader's MATLAB sentence index. Whole runs of sentences are missing for some readers, and they line up with blocks of about 50 described in the ZuCo papers: 0–49 (ZJS), 50–99 (ZPH, YAK), 49–99 (YRH), 100–153 (YFR) and 201–299 (YAC). This supports IDs keeping blocks together in order. It does not confirm the order within a block, or the order of the blocks in time.

## What this does and does not show

- **Shown:** for familiar readers, EEG features reveal roughly where a sentence sits in the list, and that alone can lift text retrieval well above chance on a random split.
- **Shown:** knowing a sentence's true place gives 55.10% with no EEG, so random splits of ZuCo are a weak test for retrieval.
- **Not shown:** that the CNN used this route, or how much of its score did. A 5.69-point gap remains, though its sentence interval includes 0.
- **Not shown:** that the position signal is drift. A slow change in EEG over the session is the likely reading, supported by the noise arm, but features could also respond to content that changes along the list. This test does not separate them.
- **Not shown:** anything about published models. Their splits and pools still need checking, so these scores cannot be set against theirs.
- **Limits:** 53 reused sentences, one split for the bridge, and every reader shared with training. The CNN checkpoint trained with zone sentences that the bridge did not get, which can only lower the bridge.
- **New readers:** untested.

## Integrity

- Scope audit: no failures. No v1 test or v2 test-zone sentence reached a fit or score.
- Re-predicting the v1 CNN reproduced all 766 saved ranks exactly, using its stored normalizer.
- No text entered the position fit. Normalization and alpha selection used training sentences only.
- [All 11,303 protected files unchanged](preservation_after.json).

Files: [decision.json](decision.json), [scores.json](scores.json), [random_splits.json](random_splits.json), [sentence_id_check.json](sentence_id_check.json), [leakage_audit.json](leakage_audit.json), [plan.json](plan.json).
