# First-fixation test: plan v9

**Status: approved by the project owner on 2026-11-14.** [The config](../configs/first_fixation_v9_20261114.json) is frozen, and its SHA-256 is recorded in `scripts/run_first_fixation.py`. Any later change needs a new version.

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

The anterior channels are the 21 channels (20% of 104) nearest the front of the head in the EGI layout bundled with MNE 1.10.2: E2, E3, E4, E5, E9, E10, E11, E12, E15, E16, E18, E19, E20, E22, E23, E24, E26, E27, E118, E123 and E124.

## Decision

Each comparison needs a gap of at least 2 points and every 95% lower bound above 0, resampling sentences, blocks and readers.

| Result | Label |
|---|---|
| Beats noise and shuffled pairs, adds to gaze, survives without the anterior channels and clears the partial check | The first-fixation waveform carries sentence information beyond noise, gaze and eye channels |
| Beats noise and shuffled pairs, but fails any of the other three | The gain does not exceed gaze, eye channels or structure |
| Does not beat noise and shuffled pairs | The first-fixation result did not hold |

## Synthetic checks before freezing

Over 20 seeds each, signals that follow only fixation duration, only word length, or appear only on the anterior channels never earned the first label. A planted waveform signal earned it, no signal earned the third label, and a sentence-level signal did not earn the first. The everyday suite keeps 5 seeds for speed.

## Changes made after approval and before freezing

No EEG feature had been fitted and no score computed when these were made.

- **Anterior channels:** the list above is now fixed.
- **Stored rows:** the v6 word-row files and word embeddings are checked by hash before anything runs. A dry run of the audit found 83,798 word rows, no protected sentence and all 21 channels.
- **Gaze in the combined arms:** the draft said gaze columns would be weighted to match the gaze-only arm. The ridge model standardises every column, so that weighting has no effect. Every column, gaze included, is instead z-scored per reader inside each fold, the same way in the real and noise versions.
- **Reader resampling** now covers the adds-to-gaze and eye-channel checks as well as noise and shuffled pairs.
- **24-way within block** is impossible, because blocks hold fewer than 24 scored sentences. 2-way and 4-way are reported within block, and 2-, 4-, 10- and 24-way against all scored sentences in a release.

## Stop rule

Stop after this run, whatever the outcome. No extra bins, channels, models or settings under this plan. No final test data is scored.
