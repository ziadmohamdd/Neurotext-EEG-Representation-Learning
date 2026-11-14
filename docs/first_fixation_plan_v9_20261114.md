# First-fixation test: plan v9

**Status: draft, not approved.** Nothing has been fitted or scored. [The config](../configs/first_fixation_v9_20261114.json) is not frozen, and no run script exists yet.

## Why this test

In [word-locked test v6](research_results/word_locked_v6_20261106/REPORT.md), the first-fixation waveform scored 12.81% at 10-way against 9.72% for matched noise. It was the only input where EEG beat noise by more than 2 points on held-out blocks. It had no interval and no gaze control, and it could be eye-movement traces.

This test asks whether that result survives proper controls. It is the last decoding attempt: if it does not hold, decoding work stops.

## The main limit: reused data

The lead was found on these same 13 held-out blocks. Retesting there can show whether it survives controls, but cannot confirm it, because results chosen from one dataset tend to shrink on the next. **Even the best outcome here stays exploratory.** Confirmation needs the v2 test blocks, which can be scored only once and are your decision.

**Expected result:** the gap will probably shrink, and gaze will probably explain it. In v6, gaze alone scored 26.05%.

## Design

1. **Input:** the stored v6 word rows, read-only. No EEG file is opened and nothing is downloaded. Protected sentences were already discarded when those rows were built.
2. **Features:** mean amplitude in 50 ms bins up to 300 ms after each first fixation, 104 channels by 6 bins.
3. **Everything else as in v6:** the same blocks, buffers, targets, scaling, ridge model and scoring.

## Controls

| Arm | What it rules out |
|---|---|
| Matched noise and shuffled pairs | Results that need no word-specific EEG |
| Gaze plus EEG against gaze plus noise | EEG that adds nothing once gaze is known |
| Without the anterior channels | Eye-movement traces, which are strongest near the eyes |
| Partial check | List position, word count and reading time |

The anterior channels are the 20% of channels nearest the front of the head, in the EGI layout bundled with MNE. The exact list will be fixed before freezing.

## Decision

Each comparison needs a gap of at least 2 points and every 95% lower bound above 0, resampling sentences, blocks and readers.

| Result | Label |
|---|---|
| Beats noise and shuffled pairs, adds to gaze, survives without the anterior channels and clears the partial check | The first-fixation waveform carries sentence information beyond noise, gaze and eye channels |
| Beats noise and shuffled pairs, but fails any of the other three | The gain does not exceed gaze, eye channels or structure |
| Does not beat noise and shuffled pairs | The first-fixation result did not hold |

## Before freezing

- Owner approves this plan.
- Fix the anterior channel list and record the word-row cache hashes.
- Synthetic fixtures over 20 seeds: a planted waveform signal must earn the first label, while signals that follow only fixation duration, word length or the anterior channels must not.
- Record any change made after approval in this file.

## Stop rule

Stop after this run, whatever the outcome. No extra bins, channels, models or settings under this plan. No final test data is scored.
