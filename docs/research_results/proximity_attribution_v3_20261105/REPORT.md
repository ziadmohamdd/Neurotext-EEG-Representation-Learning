# Proximity attribution test v3: results

Run on 2026-11-05 with the frozen plan in `configs/proximity_attribution_v3_20261105.json` (SHA-256 `d6b03507fd34de056f5cadc57d16d04fb862cc966468559598f3f29126fabb8c`). Sections marked exploratory were added after the result was seen.

**Frozen result: Inconclusive for the CNN.** Near training scored 10.26% and far training 9.92% at 10-way, on familiar readers and new sentences, with chance at 10%. Near is below the 12% the plan required to rule closeness out, so the test cannot say how much closeness contributes.

**Exploratory result: the text of neighbouring sentences, with no EEG, scores 53.94% at 10-way on the v1 validation split.** This figure includes protected sentences (see Integrity), and the clean figure is [55.10% against the CNN's 21.10%](../position_bridge_v4_20261105/REPORT.md). The v1 CNN on real EEG scored 23.10% on the same queries and pool. This shows how easy that split is for anything that knows a sentence's list position. It does not show that the CNN used this route, and it is not EEG decoding.

No score here is a new accuracy claim. All sentences were reused from earlier development, and no test data was scored.

## Frozen scores

Expected 10-way accuracy, averaged within reader and then across 30 readers, on 1,543 observations.

| Arm | Near training | Far training | Gain |
|---|---:|---:|---:|
| CNN, real EEG | 10.26 | 9.92 | 0.34 |
| CNN, matched noise | 11.14 | 10.43 | 0.71 |
| CNN, shuffled pairs | 9.54 | 9.78 | -0.25 |
| Ridge, real EEG | 14.51 | 11.41 | 3.10 |
| Ridge, matched noise | 13.17 | 11.30 | 1.87 |
| Ridge, shuffled pairs | 9.13 | 10.34 | -1.21 |
| Word counts and timing, no EEG | 12.12 | 9.94 | 2.18 |
| List position, no EEG | 11.71 | 7.60 | 4.11 |
| Text of the two closest training sentences, no EEG | 20.65 | 6.53 | 14.12 |

Exact values and 2-way and 4-way scores are in [details.json](details.json).

### Decision

| Model | Gain | Reader 95% interval | Sentence 95% interval | Reading under the frozen rule |
|---|---:|---:|---:|---|
| CNN | 0.34 | -0.43 to 1.09 | -5.37 to 6.09 | Inconclusive |
| Ridge | 3.10 | 1.60 to 4.51 | -0.86 to 6.99 | Closeness is not the main cause |

The ridge label needs care. Its gain falls in the "part" range and clears the reader bound, and the label comes from the sentence bound crossing zero. With so few sentences, this test could not rule out a gain of several points in either model. The label should not be read as evidence that closeness is unimportant.

### Other frozen effects

| Effect, in points | CNN near | CNN far | Ridge near | Ridge far |
|---|---:|---:|---:|---:|
| Full minus half size | 0.59 | -0.20 | 0.59 | -0.61 |
| v1 input minus F input | 0.10 | 0.10 | -1.60 | 0.62 |

Input choice and halving the training set changed the selected CNN by under 1 point. These runs do not cover the step up to v1's 6,894 observations.

**Far training found nothing above noise.** The far CNN sat 0.51 points below far noise, and far ridge sat 0.12 above it. At distance 11 or more, neither model showed any score that depends on the real EEG waveform.

## Exploratory: what the last epoch shows

The plan reported last-epoch scores because earlier CNNs always selected the first epoch. Selected real-EEG epochs here were again early (0 to 4), so most selected models had barely trained. Stopping 8 epochs later, as early stopping does, gives a different picture:

| CNN, last epoch | Near | Far | Gain | Reader 95% interval | Sentence 95% interval |
|---|---:|---:|---:|---:|---:|
| Real EEG | 13.68 | 8.56 | 5.12 | 3.82 to 6.40 | 0.19 to 10.12 |
| Matched noise | 12.21 | 8.94 | 3.27 | 1.33 to 5.25 | 0.29 to 6.32 |
| Shuffled pairs | 11.23 | 9.70 | 1.53 | 0.16 to 2.87 | -3.29 to 6.20 |

- The near gain appears with real training, but noise inputs produce most of it too. Real minus noise is 1.85 points, with intervals of -0.38 to 4.09 by reader and -2.85 to 6.82 by sentence.
- Far real stays below far noise, by 0.38 points.
- This points towards a gain that needs little from the real waveform, but the test was not designed to show it. The last epoch was not a planned selection rule, and arms stopped at different epochs.

## Exploratory: neighbouring text on the v1 split

Each v1 validation query was predicted from the text of the closest training sentences by sentence ID. Nothing was trained, and no EEG or test sentence was scored. Queries and pool match the v1 CNN (1,164 observations, 81 sentences).

| Predictor | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| v1 CNN, real EEG | 66.42 | 42.59 | 23.10 | 12.56 |
| Text of the closest training sentence, no EEG | 76.76 | 59.92 | 44.59 | 33.42 |
| Text of the two closest training sentences, no EEG | 83.30 | 68.85 | 53.94 | 42.43 |

The 10-way interval for the two-sentence row is 50.77 to 57.26 by reader and 44.37 to 63.16 by sentence.

Checks on this result:

- **No overlap:** training and validation share no sentence ID or sentence key.
- **Scoring works:** using two random training sentences instead of the closest gives 9.78% over 20 seeds (range 8.99 to 11.34), which is chance.
- **Not only near-copies:** with queries limited to those below 0.5 word overlap with their neighbours, the score is 53.00% (77 sentences). Below 0.3 it is 38.69% (49 sentences). Below 0.5 embedding cosine it is 39.89% (59 sentences). Median word overlap is 0.22. The pool stays the same in these subsets.

### Why v3 understated this

v3's own text control reached only 20.65% near. Its held-out sentences sit in clusters of 7, so many distractors are neighbours of the query and share its topic, and queries inside a cluster are up to 4 positions from training. In v1, held-out sentences are scattered, and every query has a training sentence at distance 1 or 2. So v3's near condition is much weaker than v1's closeness and cannot be used to size v1's advantage. This limit was not in the plan and only became clear from the control.

## What this does and does not show

- **Shown:** on a random sentence split of ZuCo, position plus neighbouring text is enough to reach a retrieval score more than double the v1 CNN's.
- **Not shown:** that the CNN reached its 23.10% this way. That needs EEG to reveal list position well enough, which is the next test.
- **Not explained:** the fall from 23.10% to about 10%. Near runs with v1 input reached 10.36% selected and 12.37% at the last epoch, on about a third of v1's training data and a weaker near condition.
- **Assumption:** sentence IDs were taken as list positions. The topic match between neighbours supports them being in text order, but presentation order remains unverified.
- **Limits:** one CNN seed per fold, reused sentences, and 81 v1 sentences. Other work's splits and pools differ, so this score cannot yet be set against published results.
- **New readers:** untested. All readers here were in training.

## Integrity

- Leakage audit: no failures. Worst Jaccard 0.39, containment 0.73, cosine 0.78.
- [All 11,119 protected files unchanged](preservation_after.json). v3 finished with exit code 0.
- v2 test blocks and their buffers were excluded from every frozen v3 arm. v1 test sentences were not scored.
- **Correction:** this report first said those blocks were excluded throughout, which was wrong for the exploratory neighbouring-text check. It scored all 81 v1 validation sentences, including 17 in v2 test blocks and 11 in their buffers, and used the text of zone sentences in training as predictions. That check used text only, with no EEG and no model. Its 53.94% therefore includes protected sentences, and [v4](../../position_bridge_plan_v4_20261105.md) recomputes it without them.
- v2 test sentences are also not fully fresh: v1 fitted or scored many of them before v2 chose its test blocks.
- Breakdown scripts: `scripts/analyze_v3_details.py`, `scripts/analyze_v1_neighbour_text.py` and `scripts/check_v1_neighbour_text.py`.

The next step is in the [v4 plan](../../position_bridge_plan_v4_20261105.md).
