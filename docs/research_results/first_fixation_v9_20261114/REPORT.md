# First-fixation test v9: results

Run on 2026-11-14 with the frozen plan in `configs/first_fixation_v9_20261114.json` (SHA-256 `ce55bbfa603ca82019920b0622ace6aa8feac1dfbbcb06ce8a9282b4703c5b79`).

**Result: the first-fixation result did not hold.** The waveform again scored 12.81% at 10-way within a held-out block, against 9.72% for matched noise, reproducing v6 exactly. With intervals, that 3.08-point gap is not clear, because its range includes 0 for sentences, blocks and readers.

**Gaze explains the score.** Gaze and word properties with no EEG scored 26.05%. Adding the real waveform to gaze gave 24.39%, and adding noise gave 24.29%, so the waveform added 0.10 points.

This was the last decoding attempt, and by the stop rule decoding work ends here. These are exploratory scores on the same blocks where the lead was found, with no final test data.

## Primary scores

Expected 10-way within the query's own held-out block, on 255 sentences across 13 blocks, all readers averaged, one equal-weight score per sentence. Chance is 10%.

| Arm | Score | ZuCo 1.0 | ZuCo 2.0 |
|---|---:|---:|---:|
| Real waveform | 12.81 | 10.85 | 14.18 |
| Matched noise | 9.72 | 10.81 | 8.96 |
| Shuffled pairs, mean of 10 | 10.11 | 11.75 | 8.96 |
| Gaze and word properties, no EEG | 26.05 | 26.33 | 25.86 |
| Gaze plus real waveform | 24.39 | 25.64 | 23.51 |
| Gaze plus noise waveform | 24.29 | 25.50 | 23.44 |
| Real, anterior channels removed | 11.89 | 9.83 | 13.33 |
| Noise, anterior channels removed | 7.35 | 6.75 | 7.77 |
| Real, position, words and reading time removed | 12.61 | 11.11 | 13.66 |
| Noise, the same removed | 10.27 | 11.22 | 9.61 |

## Decision

Each comparison needs a gap of at least 2 points and every 95% lower bound above 0.

| Comparison | Points | Sentence 95% | Block 95% | Reader 95% | Clear |
|---|---:|---:|---:|---:|---|
| Real minus noise | 3.08 | -1.50 to 7.79 | -1.91 to 9.29 | -1.51 to 5.32 | No |
| Real minus shuffled pairs | 2.70 | -0.72 to 6.29 | -1.35 to 7.20 | -0.41 to 4.42 | No |
| Gaze plus real minus gaze plus noise | 0.10 | -1.56 to 1.72 | -1.82 to 2.37 | -1.74 to 1.48 | No |
| Anterior removed, real minus noise | 4.54 | 0.75 to 8.40 | 1.29 to 8.68 | -0.96 to 5.88 | No |
| Partial check, real minus noise | 2.34 | -2.47 to 7.22 | -2.85 to 8.43 | | No |

The waveform beat neither noise nor shuffled pairs clearly, which gives the frozen label. Every other check failed too.

- **Gaze is the main finding:** once gaze is known, the waveform adds nothing measurable. Its best case is under 2 points.
- **Removing the channels nearest the eyes did not remove the gap:** without them, real still beat noise by 4.54 points, clear for sentences and blocks but not for readers. So the gap is not mainly from those channels. That does not show it is meaning, because the gaze check already explains it.
- **Noise is unusually low without the anterior channels:** 7.35%, below chance. That widens the gap, and is one reason not to read much into it.

## Secondary results

None of these changes the label. No intervals were computed.

**By time after fixation start, 10-way within block:**

| Bin | 0–50 ms | 50–100 | 100–150 | 150–200 | 200–250 | 250–300 |
|---|---:|---:|---:|---:|---:|---:|
| Real | 11.71 | 12.04 | 13.57 | 12.62 | 9.93 | 9.72 |
| Noise | 11.41 | 10.02 | 10.02 | 8.22 | 10.38 | 10.52 |

The gap sits at 50 to 200 ms, where the brain's early visual response to a newly fixated word appears. Later bins are often empty, because many fixations end before 200 ms. This pattern fits a visual response that varies with word length or contrast, which gaze and word properties already capture.

**By number of readers averaged:**

| Readers | 1 | 2 | 4 | 8 | All |
|---|---:|---:|---:|---:|---:|
| Real | 10.29 | 11.48 | 11.38 | 11.86 | 12.81 |
| Noise | 10.13 | 10.02 | 10.43 | 9.68 | 9.72 |

With one reader, the closest to real use, real and noise are both at chance.

**Other pools:** within block, 2-way scored 52.72% real and 50.50% noise, and 4-way 27.67% and 24.54%. Against all scored sentences in a release, real and noise both fall far below chance (about 31–34% at 2-way), as they did in v6. That pool combines models fitted on different folds, which is the likely cause, and it is not a meaningful score.

## What this does and does not show

- **Shown:** on held-out blocks, the first-fixation waveform adds nothing to gaze and word properties in this setup.
- **Shown:** the v6 lead does not survive intervals, even before the gaze check.
- **Not shown:** that the EEG holds no word information. This is one linear model on 50 ms bin means, on reused data. [Raghavan et al. 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205611/) found that fixation-locked EEG on ZuCo tracks word features, which fits the early visual gap seen here.
- **Limits:** the lead was chosen on these blocks, so this result can only weaken it, not confirm it. The anterior list is a rough cut of 21 channels.

## Integrity

- **No EEG file was opened:** the test used the stored v6 word rows, checked by hash.
- **Leakage audit:** no failures. No protected sentence is in the stored rows, and the fold checks matched v6.
- **Protected data:** [2,171 prior result, manifest and cache files were unchanged](preservation_after.json).
- **Tests:** 380 passed and 2 were skipped before freezing. Over 20 seeds each, duration-only, length-only and anterior-only synthetic signals never earned the first label.

Files: [decision.json](decision.json), [secondary.json](secondary.json), [leakage_audit.json](leakage_audit.json), [plan.json](plan.json). [The plan](../../first_fixation_plan_v9_20261114.md) lists changes made before freezing.
