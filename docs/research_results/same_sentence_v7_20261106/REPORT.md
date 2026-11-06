# Same-sentence test v7: results

Run on 2026-11-06 with the frozen plan in `configs/same_sentence_v7_20261106.json` (SHA-256 `5810ce7d4bf35adac3b3a5052c146c44460c357a18ae5670a44851dc52f38471`).

**Result: no same-sentence match found with these features.** Across ZuCo 1.0 and 2.0, a sentence's averaged EEG found its own copy 28.18% of the time at 10-way. That is above chance at 10%, but matched noise scored 26.68%, so EEG-specific content did not show.

**The match looks structural.** List position alone scored 54.39%, and word count with reading time scored 52.74%, both with no EEG. The shared sentences also sit in a similar order in both lists, which the plan had not expected (correlation 0.70).

These are exploratory scores on 48 sentences, with no final test data. A null here is weak evidence, and it covers these features only. The test does not show that the EEG holds no meaning.

## Primary scores

Expected 10-way across releases, mean of both directions, on 48 sentences, one equal-weight score per sentence. Nothing was fitted.

| Arm | Score |
|---|---:|
| Chance | 10.00 |
| Real EEG | 28.18 |
| Matched noise | 26.68 |
| Re-paired sentences, mean of 1000 | 9.96 |
| Word count and reading time, no EEG | 52.74 |
| List position, no EEG | 54.39 |
| Real, position, words and reading time removed | 18.40 |
| Noise, position, words and reading time removed | 16.06 |

## Decision

Each row needs a gap of at least 2 points and every 95% lower bound above 0.

| Real EEG minus | Points | Sentence 95% interval | Reader 95% interval | Clear |
|---|---:|---:|---:|---|
| Matched noise | 1.50 | -4.98 to 8.20 | -4.25 to 9.47 | No |
| Re-paired sentences | 18.23 | 9.70 to 27.60 | 6.20 to 21.95 | Yes |
| Word count and reading time | -24.56 | -36.32 to -12.59 | | No |
| List position | -26.21 | -38.16 to -13.35 | | No |
| Partial check, real minus noise | 2.34 | -2.46 to 7.15 | -4.05 to 6.56 | No |

Real EEG did not beat noise, which gives the frozen label.

- **Why noise matches too:** matched noise keeps each recording's channel amplitudes, so it keeps slow changes across the session and any link between amplitude and sentence length. What is left for real EEG beyond that is the 1.50-point gap, and its interval includes 0.
- **The partial check is still above chance:** 18.40% real and 16.06% noise. It removes only linear effects, so something structural remains in both. This means that removing these effects did not fully clean the comparison.
- **How large an effect this leaves open:** the partial gap's interval tops out at about 7 points, so a modest content effect is not excluded.

## An assumption that did not hold

The plan treated these as the same sentence at unrelated points in two lists. Between releases, list position correlates at 0.70, word count at 1.00 and mean reading time at 0.98. So the two copies share position as well as words, and this test separates content from recording block less cleanly than intended. The position arm and the partial check do cover this, which is why the label did not change.

## Secondary results

None of these changes the label. No intervals were computed.

| Check | Real EEG | Matched noise |
|---|---:|---:|
| 2-way | 73.40 | 68.77 |
| 4-way | 49.75 | 45.61 |
| 24-way | 16.29 | 16.68 |
| 48-way | 9.38 | 11.46 |
| Same-sentence cosine minus other-sentence cosine | 0.237 | 0.213 |
| Time-binned input, 10-way | 12.75 | 5.70 |
| Time-binned input, partial check, 10-way | 11.72 | 6.36 |

- **Time-binned input:** this is the only input where real EEG beat noise by more than 2 points (7.05, or 5.36 after the partial check). Treat it as a loose end, not a finding. It has no interval and is one of several side checks, and noise sitting well below chance suggests the comparison is unstable. It could also reflect eye movements, as these bins follow the time course of the sentence. The stop rule means it was not explored further.
- **Directions:** ZuCo 1.0 to 2.0 scored 28.52%, and 2.0 to 1.0 scored 27.84%.

### Block effect

Within each release, the mean cosine of sentence pairs within 25 list positions, minus farther pairs. None of 1000 shuffled lists reached any of these values.

| Release | Sentences | Real | Noise | Real, length removed | Noise, length removed | Shuffled list, 95th percentile |
|---|---:|---:|---:|---:|---:|---:|
| ZuCo 1.0 | 186 | 0.542 | 0.515 | 0.426 | 0.390 | 0.013 |
| ZuCo 2.0 | 233 | 0.336 | 0.310 | 0.303 | 0.277 | 0.012 |

**Sentences near each other in the list look strongly alike, and noise shows nearly the same effect.** So this closeness sits largely in amplitude patterns across channels that noise keeps, not in fine EEG detail. These numbers are on all usable sentences and the cosine gaps above are on 48, so the two should not be compared as exact effect sizes.

## What this does and does not show

- **Shown:** with these features, EEG shows no same-sentence match beyond matched noise across releases. List position and length explain more than EEG does.
- **Shown:** closeness in the list shapes these features strongly in both real EEG and noise.
- **Not shown:** that recording block rather than content explains published scores. This test does not use a published model.
- **Not shown:** anything about the movie-review gap. Movie reviews are not in the normal-reading data used here.
- **Limits:** 48 sentences and no block intervals. Readers and recording setup differ across releases, and list order is assumed from sentence IDs. Whether any person is in both releases remains unaudited. On synthetic fixtures, a match below about 25% could be missed.

## Integrity

- **No EEG file was opened:** the test used v5's stored features, and their hash matched the plan.
- **Leakage audit:** no failures. The 230 protected sentences were excluded, along with any text with a protected copy, and nothing was fitted.
- **Reuse:** these 48 sentences were never scored as held out in v5 or v6, but they were available for fitting there. The v5 features themselves were computed before this test was designed.
- **Protected data:** [2,117 prior result and manifest files were unchanged](preservation_after.json). Raw signal files were not hashed this time, because none was opened.
- **Tests:** 350 passed and 2 were skipped, after the run. Before freezing, no-signal, word-count-only, duration-only and position-only fixtures earned the first reading on 0 of 20 seeds each.

Files: [decision.json](decision.json), [secondary.json](secondary.json), [leakage_audit.json](leakage_audit.json), [plan.json](plan.json). [The plan](../../same_sentence_plan_v7_20261106.md) lists changes made before freezing.
