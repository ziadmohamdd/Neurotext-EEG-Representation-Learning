# Word-locked test v6: results

Run on 2026-11-06 with the frozen plan in `configs/word_locked_v6_20261106.json` (SHA-256 `3f4053515a3128dfc3269ad44e8eb6ca2cc6aa6cd4ad1a326112b323b7fca8ff`).

**Result: no sentence signal found with word-locked features.** EEG locked to each fixated word scored 14.55% at 10-way within a held-out block. That is below matched noise at 16.25%, and far below gaze and word properties with no EEG at 26.05%. Chance is 10%.

**The gain above chance looks structural.** Removing list position, word count, reading time and fixated-word count from the predictions took real EEG to 10.02%, which is chance.

These are exploratory scores on reused sentences, with no final test data. Scoring here centres each pool, so they are not on exactly the same scale as v5. They cover these features and a linear model, and do not prove that the EEG holds no meaning.

## Primary scores

Expected 10-way within the query's own block, on 255 sentences across 13 held-out blocks, one equal-weight score per sentence.

| Arm | Score | ZuCo 1.0 | ZuCo 2.0 |
|---|---:|---:|---:|
| Chance | 10.00 | 10.00 | 10.00 |
| Real word-locked EEG, averaged | 14.55 | 16.65 | 13.07 |
| Matched noise, averaged | 16.25 | 17.29 | 15.53 |
| Shuffled pairs, mean of 10 | 10.37 | 10.16 | 10.51 |
| Gaze and word properties, no EEG | 26.05 | 26.33 | 25.86 |
| List position, no EEG | 7.69 | 8.91 | 6.84 |
| Real, position, words and reading time removed | 10.02 | 10.00 | 10.03 |
| Noise, position, words and reading time removed | 11.98 | 12.26 | 11.78 |

## Decision

Each row needs a gap of at least 2 points and every 95% lower bound above 0.

| Real EEG minus | Points | Sentence 95% interval | Block 95% interval | Reader 95% interval | Clear |
|---|---:|---:|---:|---:|---|
| Matched noise | -1.70 | -6.43 to 2.81 | -5.83 to 2.22 | -4.14 to 1.52 | No |
| Shuffled pairs | 4.18 | 0.79 to 7.88 | 1.46 to 6.55 | 0.81 to 5.57 | Yes |
| Gaze and word properties | -11.51 | -16.60 to -6.34 | -16.14 to -6.97 | | No |
| List position | 6.86 | 2.45 to 11.36 | 4.07 to 9.90 | | Yes |
| Partial check, real minus noise | -1.96 | -5.81 to 1.73 | -5.95 to 1.99 | | No |

Real EEG did not beat noise, which gives the frozen label.

- **Why shuffled pairs is clear but noise is not:** the noise keeps each observation's average EEG and fixation pattern, and removes only word-specific EEG. Real EEG beating shuffled pairs shows the pipeline finds something. Failing to beat this noise means that word-specific EEG added nothing measurable on top.
- **Gaze is the strongest arm:** it beats real EEG by 11.51 points, and it uses no EEG at all. This matches the [earlier word pretraining result](../word_pretrain_v1_20261101/), where gaze-derived information beat EEG.
- **How large a gain this leaves open:** real minus noise tops out at about 2 to 3 points, so a small word-specific effect is not excluded.

## Does averaging help?

Expected 10-way within block. Both arms rise with readers, and noise stays ahead at every count. One reader is the closest to real single-recording use.

| Readers averaged | 1 | 2 | 4 | 8 | All |
|---|---:|---:|---:|---:|---:|
| Real EEG | 11.72 | 11.06 | 12.56 | 12.73 | 14.55 |
| Matched noise | 11.96 | 12.48 | 13.55 | 14.04 | 16.25 |

## Secondary results

None of these changes the label. No intervals were computed unless shown.

| Check | Real EEG | Matched noise | Other |
|---|---:|---:|---|
| Unchanged sentence-embedding targets, within block | 12.70 | 14.00 | Gaze 12.79, position 8.32 |
| Direct word test, 10-way among a block's word types | 12.61 | 10.97 | Gaze 20.54 |
| First-fixation waveform, within block | 12.81 | 9.72 | |
| New readers and new sentences, within block | 13.22 | 13.31 | Shuffled pairs 10.00 |

- **First-fixation waveform:** this is the only input where real EEG beat noise by more than 2 points (3.09). Treat it as a loose end, not a finding. It has no interval, no gaze or partial check of its own, and it is one of several side checks. Eye movements leave their own electrical traces in EEG and relate to word length, so this could be gaze again. The stop rule means it was not explored further.
- **Direct word test:** real EEG is 1.64 points above noise and well below gaze, so word properties explain more than EEG here too.
- **New readers:** 15 readers fit and the other 15 were scored. Real minus noise was -0.09 points (sentence interval -4.47 to 4.26). Real minus shuffled pairs was 3.22 points (sentence interval 0.07 to 6.66).
- **Seen word types:** 65.05% of held-out word rows have a word type that also occurs in that fold's training rows.

Against all scored sentences in a release, real EEG and noise fall below chance, while gaze stays above it:

| Arm, release | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| Real EEG, ZuCo 1.0 | 45.71 | 19.16 | 6.10 | 2.07 |
| Real EEG, ZuCo 2.0 | 40.83 | 18.49 | 6.74 | 2.53 |
| Matched noise, ZuCo 1.0 | 48.72 | 22.22 | 6.95 | 2.03 |
| Matched noise, ZuCo 2.0 | 42.22 | 18.90 | 6.54 | 2.57 |
| Gaze and word properties, ZuCo 1.0 | 60.75 | 36.40 | 18.38 | 8.39 |
| Gaze and word properties, ZuCo 2.0 | 66.52 | 40.55 | 18.07 | 7.22 |

These pool predictions from models fitted on different folds, which may be why they fall below chance. This was not tested, so they should not be read as a real anti-signal.

## What this does and does not show

- **Shown:** with supplied band features and a linear model, word-locked EEG on held-out blocks does not beat noise that keeps sentence-level EEG and the fixation pattern. Gaze and word properties with no EEG score far higher.
- **Shown:** the gain above chance mostly goes away when position, word count, reading time and fixated-word count are removed.
- **Not shown:** that ZuCo EEG holds no meaning. The primary input starts at 4 Hz, and classic meaning-related brain responses sit largely below that. A nonlinear model might also find more, though nothing here suggests it.
- **Limits:** 255 reused sentences and 13 blocks, sentence meaning approximated by a mean of word embeddings, and list order assumed from sentence IDs. Block-level scores vary widely (real 5.4 to 24.6, noise 3.4 to 25.8). Alpha choices also varied between folds, from 0.1 to 100,000.

## Integrity

- **Crash and rerun:** the first run crashed at the scoring step, before any score was produced. A prediction had almost no length left after the partial check, and the scoring code rejected it. Such a prediction now counts as a miss (commit `43022b0`), and the same frozen plan was rerun. Three sentences were affected in the real arm and three in the noise arm, in the partial check only.
- **Leakage audit:** no failures. Lowest list distance from held-out to training was 11, with token Jaccard at most 0.35, containment at most 0.70 and text cosine at most 0.82.
- **Protected data:** v1 test sentences, v2 test blocks and their buffers were not stored, fitted or scored. Their source word fields were skipped while reading the files, and [all 11,336 protected files were unchanged](preservation_after.json).
- **Tests:** before the first run, 333 tests passed and 2 were skipped. Position-only, length-only, frequency-only and duration-only synthetic signals each earned the first reading on 0 of 20 seeds. After the crash fix, the word-locked tests passed again on 5 of those seeds.
- **Words:** 83,798 usable word rows, 48,996 in scored sentences, from 2,501 word types.

Files: [decision.json](decision.json), [secondary.json](secondary.json), [selection.json](selection.json), [word_counts.json](word_counts.json), [leakage_audit.json](leakage_audit.json), [plan.json](plan.json). Word rows are kept out of git in `data/processed/word_locked_v6_20261106/`.
