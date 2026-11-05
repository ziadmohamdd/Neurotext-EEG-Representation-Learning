# Position bridge test: plan v4

**Status: proposed, not approved, nothing built or run.** Thresholds and settings get frozen with a config hash only after approval.

This plan does not try to raise accuracy. It tests whether v1's score could come from EEG revealing list position, since [v3](research_results/proximity_attribution_v3_20261105/REPORT.md) found that neighbouring text alone scores 53.94% at 10-way on the v1 split.

## Question

Can a model that maps EEG only to list position, and then copies the text of training sentences at that position, match the v1 CNN's 23.10% at 10-way?

## Parts

1. **EEG-to-position model:** ridge from EEG features to list position, fitted on v1 training sentences and the same 30 readers. Arms are real EEG, matched noise and shuffled positions.
2. **Bridge score:** take the predicted position, average the text of the closest training sentences, and score with the v1 queries and pool. This model never sees text during fitting.
3. **Size of the baseline:** repeat neighbouring-text scoring over 20 random splits and several held-out fractions, with no EEG.
4. **Sentence ID check:** look for evidence of presentation order in the raw ZuCo files, and report if it cannot be confirmed.

## Proposed reading

| Result | Reading |
|---|---|
| Real bridge reaches at least 21.10% and beats shuffled | Position alone could account for v1's score. The paper route becomes the lead |
| Real bridge stays clearly below this, but beats shuffled | Position explains part of it, and the rest stays open |
| Real bridge does not beat shuffled | This route is not supported. The 53.94% stays a warning about the split and nothing more |

Noise arms show how much of any bridge score needs the real waveform. A score matched by noise would point to amplitude differences across a session, not to reading content.

## Scope and cost

- **Data:** existing data and text cache, with no download.
- **Protected:** v1 test sentences, v2 test blocks and held-out new readers are not scored. v1 validation is reused, so this is exploratory and not independent confirmation.
- **Time:** about an hour to build and test on synthetic data, then minutes to run on CPU.

## Afterwards

- **Published splits:** checking how SemKey, GLIM and ETER treat list position needs their code or split files, so I will list exact sources and sizes for approval first.
- **Literature:** the [first check](paper_route_literature_check_20261105.md) needs a systematic search and a supervisor review before this is called new.
- **New readers:** the plan is still owed. v3 found nothing above noise at a distance even for familiar readers, so calibration has no known signal to transfer yet. I recommend writing it after this test.
