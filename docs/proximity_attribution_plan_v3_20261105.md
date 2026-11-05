# Proximity attribution test: plan v3

**Status: approved by the project owner on 2026-11-05.** SHA-256 of [the config](../configs/proximity_attribution_v3_20261105.json): `d6b03507fd34de056f5cadc57d16d04fb862cc966468559598f3f29126fabb8c`. Both files are frozen. Any later change needs a new version.

This plan does not try to raise accuracy. It explains why accuracy fell, so that later accuracy work targets the right cause.

## Why v3 exists

The CNN's 10-way validation score fell from 23.10% in v1 to 10.02% in v2. v2 changed three things at once, so the drop cannot be assigned to any one of them:

| Difference | v1 | v2 |
|---|---|---|
| Distance from training sentences in the list | Often next to a held-out sentence | At least 11 positions away |
| Training data | 6,894 observations | 4,487 observations |
| Input | No notch, Cz kept, 8 seconds, shared normalizer | 50 Hz notch, no Cz, up to 30 seconds, per-reader normalizer |

The two pilots also scored different sentences.

There is a further clue. In v1, the real-EEG CNN's best epoch was 10 and validation loss improved. In v2, all three real-EEG runs had their best epoch at the first epoch.

## What the ZuCo papers say about order

- **ZuCo 1.0:** the [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289117/) says "sentences were presented to all subjects in identical order".
- **ZuCo 2.0:** the [paper](https://ar5iv.labs.arxiv.org/html/1912.00903) says the "order of blocks and sentences within blocks was identical for all subjects". Normal-reading blocks of about 50 sentences alternate with task blocks, with a short break after each block.

So each sentence is tied to one point in the session for every reader, and a random split can place training sentences right next to held-out ones in time.

Two things remain unverified: that sentence IDs equal presentation positions, and where the block boundaries and breaks fall. The strong link between EEG and list position supports the first but does not prove it.

## Question and hypothesis

With the same held-out sentences, input and training size, how much does training on sentences close in the reading order raise accuracy, compared with training only on distant sentences?

**Hypothesis:** near training beats far training by at least 5 percentage points at 10-way. This would mean closeness explains most of v1's score.

## Design

Only the distance between training and held-out sentences changes.

```
List:   [ held-out 7 ][ near 5 ][ gap 5 ][ far ... ][ gap 5 ][ near 5 ][ held-out 7 ]
Lag:                    1-5       6-10     11+        6-10     1-5
```

- **Held-out clusters:** 7 contiguous sentences, starting every 42 positions in each list.
- **Folds:** 3, starting at offsets 0, 14 and 28, so no sentence is held out twice.
- **Far condition:** all sentences at list distance 11 or more.
- **Near condition:** all sentences at list distance 1 to 5, topped up with far sentences until each release matches the far condition's sentence count. Mixing near and far resembles v1, where training also had both.
- **Unused:** sentences at distance 6 to 10.
- **Settings:** in each fold and release, every third cluster selects settings and the rest are scored, so no score comes from the sentences that chose its checkpoint.

### Sizes per fold

Measured from sentence IDs and observation counts only, with no EEG read:

| Release | Selection sentences | Scored sentences | Near sentences | Training sentences, each condition |
|---|---:|---:|---:|---:|
| ZuCo 1.0 | 7 | 14–15 | 28–34 | 71–80 |
| ZuCo 2.0 | 7–13 | 17–24 | 37–49 | 77–100 |

Each fold trains on about 2,100–2,600 observations, well below v1's 6,894, and 1,543 observations are scored across the folds. Near and far use the same number of sentences. Their observation counts differ by at most 7 per fold at full size.

### Changes made after approval and before freezing

These came from building the test on sentence IDs and text, before any EEG was read:

- **Extra exclusion:** the audit found held-out sentences sharing at least 80% of their words with training sentences, including neighbouring sentences about the same person. Every sentence in such a pair now joins the excluded list.
- **Selection score:** selection pools of about 7 sentences cannot give a 10-way score, so ridge settings are chosen on expected 4-way accuracy.
- **24-way dropped:** ZuCo 1.0 scored pools hold only 14 or 15 sentences.

### Runs

| Model | Input | Size | Conditions | Arms | CNN runs across 3 folds |
|---|---|---|---|---|---:|
| CNN | F | Full | Near, far | Real, noise, shuffled pairs | 18 |
| CNN | F | Half | Near, far | Real | 6 |
| CNN | v1 | Full | Near, far | Real | 6 |

Ridge runs the same arms. Word counts and timing, list position, and text from the two closest training sentences run for both conditions with no EEG.

The size arms measure the data effect. The v1-input arms measure the preprocessing effect.

Before any training, I will also bin the saved v1 validation scores by each query's distance to its closest training sentence. This costs nothing but is weaker evidence, since most v1 queries have close neighbours.

## Measurement and decision

**Primary score:** expected 10-way accuracy of the CNN on real EEG with input F, against scored sentences from the same fold and release, averaged within reader and then across 30 readers. Chance is 10%.

**Gain:** near minus far on the same queries, with 10,000 bootstrap resamples of readers and of sentences.

| Result | Reading |
|---|---|
| Gain at least 5 points, both 97.5% lower bounds above 0, far within 2 points of far noise | Closeness explains most of the score |
| Gain from 2 up to 5 points, both lower bounds above 0 | Closeness explains part of the score |
| Gain below 2 points or a lower bound at or below 0, with near at least 12% | Closeness is not the main cause |
| Gain below 2 points or a lower bound at or below 0, with near below 12% | Inconclusive, as the smaller training set may hide the effect |
| Anything else | Inconclusive |

The size and input effects are reported with intervals but have no pass mark. All of this is exploratory, and no result here is a new accuracy claim.

## What happens next

| Outcome | Next plan |
|---|---|
| Closeness explains most or part | Literature check, then write up the benchmark route and test SemKey-style splits if feasible |
| Closeness is not the main cause | Look at whichever of size or input shows the larger effect |
| Inconclusive | Report it and decide with the owner; more reading data becomes the leading option |

New-reader work follows under its own plan in every case.

## Limits to keep visible

- **Cause of a gain:** a near gain cannot by itself separate slow EEG drift from real decoding helped by similar neighbouring topics. The list-position and neighbouring-text controls bound this but do not settle it.
- **Training size:** less training data than v1 may shrink every score.
- **Reused data:** v2's validation sentences can appear here, so this is not independent confirmation.
- **Seeds:** each fold has one CNN seed, so seed and fold differences are mixed together. Near and far share the seed in each fold, so their comparison stays paired.
- **Checkpoints:** if the first epoch is selected again, near and far could look alike because neither trained for long. The last-epoch score is reported for this reason.
- **Readers:** the readers are shared, so this says nothing about new people.

## Leakage safeguards

The runner writes an audit and refuses to start if any check fails:

- No v2 test sentence or test buffer sentence appears in any fit or score.
- No sentence group, observation or signal file appears in both training and held-out sets.
- No held-out sentence has close wording to any training sentence (token Jaccard below 0.7, containment below 0.8) or a near-identical embedding (cosine below 0.9).
- Far training stays more than 10 positions from every held-out sentence in its fold.
- Near and far match in sentence count per release and fold.
- Selection clusters are never scored.

Normalization and weights fit on training rows only. Existing datasets, targets, checkpoints and results stay unchanged.

## Work and time after approval

1. Write the split, the audit, the v1 input and the runner, and test them on synthetic data only.
2. Record exact sizes from sentence IDs, then freeze the config and its hash.
3. Run the no-training v1 analysis.
4. Run the 30 CNN runs and the ridge and control runs.

v2's CNN runs took 8–12 minutes on 4,487 training observations. With about half that data, I estimate 4–6 minutes per run, so **about 2–3 hours** of GPU time plus setup. Outputs should need under 100 MB on C:, which has 13 GB free. No download is needed.
