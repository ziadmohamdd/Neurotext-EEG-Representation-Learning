# Averaged-EEG test v5: results

Run on 2026-11-06 with the frozen plan in `configs/averaged_eeg_v5_20261106.json` (SHA-256 `d006d67e421de6f01191c159856bc1a182be8a15e7696920dac5d575aac68bf6`).

**Result: no sentence signal found with these features.** Averaging EEG across readers scored 11.25% at 10-way within a held-out block, against 12.13% for matched noise and 12.07% for timing with no EEG. Chance is 10%.

These are exploratory scores on reused sentences, with no final test data. They come from whole-sentence feature summaries, so they do not rule out a word-locked signal or prove that the EEG holds no meaning.

## Primary scores

Expected 10-way within the query's own block, on 255 sentences across 13 held-out blocks, one equal-weight score per sentence.

| Arm | Score | ZuCo 1.0 | ZuCo 2.0 |
|---|---:|---:|---:|
| Chance | 10.00 | 10.00 | 10.00 |
| Real EEG, averaged | 11.25 | 11.81 | 10.86 |
| Matched noise, averaged | 12.13 | 13.47 | 11.19 |
| Shuffled pairs, mean of 10 | 10.18 | 10.51 | 9.96 |
| Timing and word count, no EEG | 12.07 | 12.24 | 11.95 |
| List position, no EEG | 9.53 | 9.97 | 9.22 |
| Real, position and timing removed | 10.57 | 11.96 | 9.59 |
| Noise, position and timing removed | 13.04 | 15.22 | 11.52 |

## Decision

Each row needs a gap of at least 2 points and every 95% lower bound above 0.

| Real EEG minus | Points | Sentence 95% interval | Block 95% interval | Reader 95% interval | Clear |
|---|---:|---:|---:|---:|---|
| Matched noise | -0.88 | -3.01 to 1.19 | -3.32 to 0.95 | -1.91 to 0.92 | No |
| Shuffled pairs | 1.07 | -1.06 to 3.23 | 0.26 to 1.92 | 0.17 to 2.02 | No |
| Timing | -0.82 | -4.13 to 2.51 | -3.61 to 2.09 | | No |
| List position | 1.72 | -1.97 to 5.44 | -2.25 to 5.56 | | No |
| Partial check, real minus noise | -2.48 | -6.67 to 1.68 | -6.63 to 1.70 | | No |

Real EEG did not beat noise, which gives the frozen label. So timing, position and the partial check could not change it.

- **How large a gain this leaves open:** real minus noise tops out at about 1 point. Real minus shuffled pairs could still be up to about 3 points, so a small effect is not excluded.
- **Noise is above chance:** it keeps channel amplitudes, and its score rises with more readers. This run does not show why. It is why noise, not chance, is the bar.

## Does averaging help?

Expected 10-way within block. A real shared signal should rise with readers, and faster than noise. It did not.

| Readers averaged | 1 | 2 | 4 | 8 | All |
|---|---:|---:|---:|---:|---:|
| Real EEG | 11.00 | 11.22 | 10.98 | 11.18 | 11.25 |
| Matched noise | 11.05 | 10.91 | 11.51 | 11.96 | 12.13 |

## Secondary results

None of these changes the label. No intervals were computed for the input and fit variants.

| Check | Real EEG | Matched noise | Other |
|---|---:|---:|---|
| Time-binned input, within block | 10.71 | 9.95 | |
| Fit on sentence means, within block | 14.22 | 12.60 | |
| New readers and new sentences, within block | 9.67 | 11.23 | Shuffled pairs 9.90 |

- **New readers:** 15 readers fit and the other 15 were averaged. Real minus noise was -1.57 points (sentence interval -4.28 to 1.06), so this is still at chance. This is the first test here of new readers and new sentences together.
- **Fit on sentence means:** this is the highest real score, 1.62 points above noise. That is below the 2-point bar, it is one of several secondary checks, and it has no interval or controls beyond noise. The stop rule means it was not explored further, and any follow-up would need its own frozen plan.

Against all scored sentences in a release:

| Arm, release | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| Real EEG, ZuCo 1.0 | 49.45 | 25.27 | 10.17 | 3.74 |
| Real EEG, ZuCo 2.0 | 49.32 | 24.28 | 10.17 | 5.05 |
| Matched noise, ZuCo 1.0 | 49.94 | 24.75 | 8.80 | 2.52 |
| Matched noise, ZuCo 2.0 | 48.61 | 23.90 | 9.64 | 4.80 |

## What this does and does not show

- **Shown:** on held-out blocks, averaging whole-sentence EEG features across readers does not separate sentences better than noise or timing, and more readers do not help.
- **Not shown:** that ZuCo EEG holds no sentence meaning. These features discard timing inside a sentence, and readers reach each word at different moments.
- **Limits:** 255 reused sentences and 13 blocks, one linear model, and list order assumed from sentence IDs. Block intervals rest on few blocks. The partial check removes linear effects only, and block-level scores vary widely.

## Integrity

- Leakage audit: no failures. Lowest list distance from held-out to training was 11, with token Jaccard at most 0.35, containment at most 0.70 and text cosine at most 0.82.
- v1 test sentences, v2 test blocks and their buffers were not parsed as EEG, fitted or scored. Their files were only hashed, and [all 11,322 protected files were unchanged](preservation_after.json).
- Before the run, 299 tests passed and 2 were skipped. Those include synthetic checks that planted signals are recovered and that duration-only and position-only signals cannot earn the first reading.
- The run took about 17 minutes on CPU, below the 1 to 2 hour estimate.

Files: [decision.json](decision.json), [secondary.json](secondary.json), [selection.json](selection.json), [leakage_audit.json](leakage_audit.json), [plan.json](plan.json). Features are kept out of git in `results/averaged_eeg_v5_20261106/`.
