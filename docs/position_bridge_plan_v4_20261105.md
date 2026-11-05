# Position bridge test: plan v4

**Status: approved by the project owner on 2026-11-05.** [The config](../configs/position_bridge_v4_20261105.json) is frozen, and its SHA-256 is recorded in `scripts/run_position_bridge.py`. Any later change needs a new version.

This plan does not try to raise accuracy. It tests whether v1's score could come from EEG revealing list position, since [v3](research_results/proximity_attribution_v3_20261105/REPORT.md) found that neighbouring text alone far outscores the v1 CNN on the v1 split.

## Question

Can a model that maps EEG only to list position, and then copies the text of training sentences at that position, match the v1 CNN at 10-way on the same queries?

## Parts

1. **EEG-to-position model:** ridge per release from EEG features to sentence ID, fitted on v1 training sentences and the same 30 readers. No text enters the fit. The alpha comes from held-back training sentences.
2. **Bridge score:** take the predicted position, average the text of the two closest training sentences, and score on the v1 validation queries that remain. Arms are real EEG, matched noise and shuffled positions (mean of 10 shuffles). v1 input is primary, and F input is secondary.
3. **Reference:** the saved v1 CNN checkpoint, re-predicted on the kept queries with no refit. Neighbouring text is scored on the same queries.
4. **Size of the baseline:** neighbouring text over 20 random splits at held-out fractions of 10%, 15% and 25%, with no EEG.
5. **Sentence ID check:** report runs of missing sentences per reader. IDs are each reader's MATLAB sentence index, so the manifests do not record presentation order directly.

## Changes made after approval and before freezing

- **Protected sentences removed:** 28 of v1's 81 validation sentences and 120 of its 486 training sentences fall in the v2 test zone. The proposal only said these would not be scored. They are now left out of every fit and score, so 53 scored sentences (766 observations) and 366 training sentences (5,262 observations) remain.
- **Reference changed:** the proposal compared with 23.10%, which includes protected sentences. The reference is now the CNN on the kept queries against the kept pool. Its stored normalizer is reused, so no protected EEG is read.
- **Prediction check:** re-predicting the CNN reproduced all 766 saved v1 ranks exactly, before the config was frozen. The run stops if under 99% match.
- **Random splits:** these now use only sentences outside v1 test and the v2 test zone.
- **Fractions:** 10%, 15% and 25%, chosen so every pool can support 10-way.

## Reading

**Beats shuffled** means a gap of at least 2 points, with both 97.5% lower bounds (readers and sentences) above 0.

| Result | Reading |
|---|---|
| Beats shuffled, and within 2 points of the CNN or above it | Position alone could account for the CNN score |
| Beats shuffled, and more than 2 points below the CNN | Position explains part of the CNN score |
| Does not beat shuffled | This route is not supported |

Noise arms show how much of any bridge score needs the real waveform. A score matched by noise would point to amplitude differences across a session, not to reading content.

## Limits to keep visible

- **Possible, not proven:** a match shows this route is enough, not that the CNN took it. The sentence correlation between the two is only suggestive.
- **Handicap:** the bridge has fewer training sentences than the CNN, since the CNN's checkpoint was trained with zone sentences. This can only lower the bridge.
- **Reused data:** v1 validation sentences have been scored before, so this is exploratory.
- **Readers:** all are shared with training, so this says nothing about new people.

## Scope and cost

No download and no final test data. Hashing and feature extraction should take minutes to about an hour on CPU, with one short GPU prediction.
