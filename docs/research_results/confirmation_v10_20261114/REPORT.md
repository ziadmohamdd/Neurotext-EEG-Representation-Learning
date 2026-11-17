# Confirmation v10 on the reserved test blocks: results

Run on 2026-11-17 with the frozen plan in `configs/confirmation_v10_20261114.json` (SHA-256 `885e285e153d0d53da8f8d6e82243dc7d9290d98c498a8eab18333546dc91f6f`). [The plan](../../confirmation_plan_v10_20261114.md) lists changes made before freezing.

**Result: all three claims were confirmed on sentences never fitted or scored before.** These were the last clean held-out sentences in ZuCo normal reading. The scores come from 85 sentences in 4 blocks, read by 29 readers, with the v5, v6 and v9 pipelines reused unchanged.

| Claim | Label | Key numbers (10-way within block, chance 10%) |
|---|---|---|
| C1. EEG does not beat matched noise | **Confirmed** | Whole-sentence 11.04 vs noise 10.10. Word-locked 13.47 vs 15.92. First fixation 12.90 vs 16.79. None clear |
| C2. Gaze beats word-locked EEG | **Confirmed** | Gaze and word properties 25.58 vs word-locked EEG 13.47: 12.12 points, sentence interval 3.96 to 20.38 |
| C3. EEG adds little to gaze | **Confirmed** | Gaze plus first-fixation EEG 25.29 vs gaze plus noise 26.65: −1.36 points. Upper ends 1.19 (sentences) and 1.44 (readers), both below the 3-point margin |

**The first-fixation lead reversed.** On development blocks it scored 12.81% against 9.72% for noise. On the fresh blocks it scored 12.90% against 16.79%.

## Primary scores

Expected 10-way within the query's own block, all readers averaged, one equal-weight score per sentence. Blocks keep 20 to 23 sentences after v1 test sentences were removed.

| Arm | Whole-sentence (v5) | Word-locked (v6) | First fixation (v9) |
|---|---:|---:|---:|
| Real EEG | 11.04 | 13.47 | 12.90 |
| Matched noise | 10.10 | 15.92 | 16.79 |
| Shuffled pairs | 9.95 | 8.99 | 10.02 |
| Timing and word count, no EEG | 8.60 | | |
| List position, no EEG | 10.64 | 8.17 | |
| Gaze and word properties, no EEG | | 25.58 | 25.58 |
| Gaze plus real | | | 25.29 |
| Gaze plus noise | | | 26.65 |
| Real, anterior channels removed | | | 11.30 |
| Noise, anterior channels removed | | | 12.44 |
| Real, partial check | 13.41 | 13.60 | 9.10 |
| Noise, partial check | 6.46 | 12.43 | 13.86 |

## Claim inputs

Points; 95% intervals from resampling sentences and readers.

| Comparison | Gap | Sentence 95% | Reader 95% | Clear |
|---|---:|---|---|---|
| v5 real minus noise | 0.94 | −0.99 to 3.04 | −1.20 to 2.81 | No |
| v6 real minus noise | −2.45 | −8.53 to 3.56 | −8.08 to 2.93 | No |
| v9 real minus noise | −3.89 | −12.09 to 4.10 | −8.72 to 3.42 | No |
| v6 gaze minus real | 12.12 | 3.96 to 20.38 | | Yes |
| v9 gaze plus real minus gaze plus noise | −1.36 | −4.20 to 1.19 | −4.26 to 1.44 | No; upper ends below 3 |

## Development against test

| Pipeline | Development real / noise | Test real / noise |
|---|---|---|
| Whole-sentence (v5) | 11.25 / 12.13 | 11.04 / 10.10 |
| Word-locked (v6) | 14.55 / 16.25 | 13.47 / 15.92 |
| First fixation (v9) | 12.81 / 9.72 | 12.90 / 16.79 |
| Gaze and word properties | 26.05 | 25.58 |

The test scores sit close to the development scores for real EEG and gaze. Matched noise moved most, which fits its being a noisier arm on 85 sentences.

## Secondary results

None of these changes a label. No intervals were computed unless shown.

- **One reader:** real and noise scored 10.85 and 9.96 (v5), 9.55 and 11.34 (v6), and 11.04 and 9.63 (v9). All are near chance.
- **New readers on new sentences:** fitting on half of each release's readers and averaging the other half gave 11.06 real vs 11.28 noise for whole-sentence features. Word-locked features gave 11.32 vs 16.89.
- **The v5 partial check:** real scored 13.41% against 6.46% for noise, a 6.94-point gap with sentence interval 0.63 to 13.62. This is not a claim in the plan, and it does not change C1, because real EEG did not beat noise before the check. It is reported because its interval is above 0. Noise falls far below chance here, as in v9's anterior check, which inflates the gap.
- **First-fixation time bins:** the 50–200 ms gap from development did not appear. In all three of those bins real scored below noise, for example 9.95 against 15.43 at 150–200 ms. At 200–250 and 250–300 ms, real was 2.26 and 2.09 points above noise. Those bins are often empty, because many fixations end before 200 ms, and they have no intervals.

## Integrity

- **Data opened for the first time:** EEG and word rows of the 85 test-block sentences. Buffers and v1 test sentences were neither fitted nor scored.
- **Inputs matched the stored data:** before any score, every rebuilt v5 feature and v6 word row outside the test blocks matched the stored values exactly.
- **Leakage audit:** no failures. Training sat at least 11 list positions from any test block.
- **Protected files:** [11,462 files were unchanged](preservation_after.json), covering earlier results, manifests, caches and EEG sources.
- **Reruns of the reused pipelines:** v5, v6 and v9 on their old layouts reproduced their saved decisions and secondary results exactly ([reproduction.json](reproduction.json)).
- **Tests:** 430 passed and 2 were skipped, with 20 seeds for the new synthetic tests.
- **Interruptions before scoring:**
  - The first attempt stopped during input preparation, because worker processes could not import a runner loaded by file path. It was fixed by running extraction in threads (commit `973941b`).
  - A second attempt stopped during extraction, most likely because the computer slept.
  - Both resumed from saved inputs. Nothing was scored until the final attempt, which scored once.
- **The v2 rule:** the frozen v2 plan said these blocks would never be scored, because no v2 model passed its gate. This plan changed that rule with the owner's approval, to test claims about controls. It added no model or setting.

Files: [claims.json](claims.json), [leakage_audit.json](leakage_audit.json), [word_counts.json](word_counts.json), [plan.json](plan.json), and each pipeline's decision and secondary JSONs in [v5](v5/), [v6](v6/) and [v9](v9/).
