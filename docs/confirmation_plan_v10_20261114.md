# Confirmation on the reserved test blocks: plan v10

**Status: draft, not approved.** Nothing here has run, no test-zone EEG has been read, and the config is not frozen. After approval, the config's SHA-256 goes into the run script, and any later change needs a new version.

## Why this test

Every block-holdout result in the paper (v5, v6, v9) comes from 255 sentences that were all used in earlier development, and the first-fixation lead was found on them. The only data never fitted or scored is the v2 test zone: 2 blocks of 25 sentences in each release, plus 10 buffer sentences on each side. This plan scores that zone once, with the pipelines and settings already frozen in v5, v6 and v9, to confirm or contradict the paper's claims on fresh sentences.

## A commitment this plan changes

The frozen v2 plan said: "Score a model and its matched controls once on test blocks only if that model passes the gate; otherwise never score test blocks." No v2 model passed, so under v2 these blocks were never to be scored.

That rule was there to stop a failed decoder from being given a second chance on test data. This plan does not do that. It tests different, pre-stated claims, mostly that EEG does *not* beat its controls, with no new model, feature or setting. The project owner approves this change, and the paper must report it, quoting the v2 rule. If a reader judges that this weakens the result, the v5, v6 and v9 results stand unchanged on their own.

## Data

| Item | Value |
|---|---|
| Scored blocks | 4: 2 in ZuCo 1.0, 2 in ZuCo 2.0, as placed by the v2 split (seed 42) |
| Scored sentences | 100 (50 per release) |
| Scored observations | 1,386, from 29 readers (v2 exact sizes) |
| Buffers | The 10 sentences on each side of every test block are neither fitted nor scored |
| Training | All 419 usable sentences outside the test zone, from both releases. These include the 255 sentences scored earlier. One fit serves all four blocks |
| Still closed | v1 test sentences stay out of every fit and score |

The test zone's EEG and word rows will be read for the first time, into a new cache. Existing caches, results and targets are not changed.

## Pipelines

Each pipeline is reused exactly as frozen. Only the layout changes, so that the four test blocks are the scored folds and the whole test zone stays out of training.

| Pipeline | Frozen in | Config SHA-256 |
|---|---|---|
| Whole-sentence features averaged across readers | v5 | `d006d67e421de6f01191c159856bc1a182be8a15e7696920dac5d575aac68bf6` |
| Word-locked band features | v6 | `3f4053515a3128dfc3269ad44e8eb6ca2cc6aa6cd4ad1a326112b323b7fca8ff` |
| First-fixation waveform, with gaze arms and anterior removal | v9 | `ce55bbfa603ca82019920b0622ace6aa8feac1dfbbcb06ce8a9282b4703c5b79` |

The settings carried over unchanged are features, alpha grids, how alpha is chosen inside training data, noise construction, shuffle counts, seeds, scoring and partial-check covariates. The text targets are the unchanged MiniLM embeddings and the existing word-embedding cache, extended with any new word types into a separate file.

## Primary score

Expected 10-way within the query's own block, all readers averaged, one equal-weight score per sentence, ties as misses. Chance is 10%. Each block has 25 sentences, so every query has 24 within-block candidates.

## Claims tested

A result is **clear** if the gap is at least 2 points and every lower bound from resampling sentences and readers is above 0. Resampling blocks is not used, because 4 blocks are too few. Block-level scores are reported as they are.

| Claim in the paper | Test | Confirmed if | Contradicted if |
|---|---|---|---|
| C1. EEG does not beat matched noise | Real minus noise, for each of v5, v6 and v9 | No pipeline is clear | Any pipeline is clear |
| C2. Gaze beats word-locked EEG | Gaze and word properties minus real word-locked EEG (v6) | Clear | Real EEG is clear over gaze |
| C3. EEG adds little to gaze | Gaze plus first-fixation EEG minus gaze plus noise (v9) | Upper ends of the sentence and reader intervals are both below 3 points | Clear |

Any other outcome for a claim is **inconclusive**. Each claim gets its own label, and the paper reports all three whatever they say. Why 3 points: with 100 sentences, the v9 interval would be about ±2.6 points, so a 2-point margin would almost always be inconclusive.

## Secondary results

None of these changes a label. All are reported.

- The list-position, timing, shuffled-pairs and partial-check arms of each pipeline.
- Each release separately.
- The score against reader count (1, 2, 4, 8 and all), real and noise.
- The v9 time bins and the v9 anterior-removal arm.
- **New readers:** fit on half of each release's readers, using the v5 and v6 split rule, and average the other half on the same test blocks.
- **Comparison with development:** each pipeline's development score next to its test score.

## Checks before running

1. **Tests on synthetic fixtures** show that:
   - no test-zone sentence or buffer enters any fit;
   - every fold's training excludes the whole test zone, not only its own block;
   - position-only and duration-only signals cannot make C1 contradicted;
   - a planted signal is found.
2. **Same results on the old layout:** run on its old layout, each reused pipeline reproduces its frozen development result exactly.
3. **Leakage audit:** it refuses to run on any failure.
4. **Hashes:** the hashes of all existing results, manifests and caches are recorded before the run and checked after it.

## Stop rule

Score once. No reruns, extra arms, settings or pipelines under this plan, whatever the outcome. After this, no clean held-out data remains in ZuCo normal reading, so any later claim would need new data.

## Risks

- **Small:** 100 sentences in 4 blocks give wide intervals, so C1 and C3 may be confirmed only loosely, or come out inconclusive.
- **Placement:** the test blocks were placed by the same seeded rule as the validation blocks, and they were checked against repeats and near-duplicates. Their position in the list differs from the earlier blocks, so drift may differ.
- **Reader 29 of 30:** one reader has no test-zone rows. This is a known v2 size.
- **Not independent of the design:** every design choice was shaped by the earlier blocks. This test confirms on fresh sentences, not on fresh readers or a fresh dataset.

## Cost

- **No download:** the EEG files are already local.
- **Run time:** extracting word rows for the test zone needs about 30 reader files. It is expected to take tens of minutes, and the full run, including the new-reader arm, up to a few hours on this laptop.
- **Before freezing:** new code is needed for the confirmation layout and a runner that calls the three frozen pipelines, plus the tests above.
