# Sentence-holdout pilot: frozen plan v1

**Status: approved by the project owner on 2026-11-05.** SHA-256 of
[the config](../configs/sentence_holdout_pilot_v1_20261105.json):
`e66ecee76d49f40c42bbffd3ac92481440c81da7b421cc0989d4a7a1edb84bda`. Both files are frozen. Any later change needs a new version.

## Question

When readers are familiar and sentences are new, does real EEG beat random noise
and word-count/timing controls?

Earlier work tested the opposite case: new readers, mostly on sentences already
seen in training. The [SemKey paper](https://arxiv.org/abs/2603.03312) uses a split
closer to this one, keeping sentences separate without holding out readers. This
pilot tests that one difference with the existing small models.

## Hypothesis and how it could fail

**Hypothesis:** with sentences held out and readers shared, real EEG reaches at
least 2 percentage points higher expected 10-way accuracy than each control.

The hypothesis fails if any part of the gate below fails on validation. Earlier
results give no reason to expect a pass: real EEG stayed near its controls in
every scope, including the earlier unseen-content runs. Those runs also held out
readers, so they did not test this setup.

## Place in the project brief

This pilot informs brief steps 4 (noise control and sentence information) and 7
(comparison and limitations). It does **not** replace step 6: generalization to
unseen subjects remains the required test, and its existing results stay reported
separately. The output must always be labelled "familiar readers, new sentences".

## Cohort and split

Existing ZuCo 1.0 and 2.0 normal-reading observations after duration QC: 9,233
observations from 30 readers. No download is needed. Prepared signals for every
observation and text embeddings for all 594 sentence keys were found on disk.

Pooling is safe here because readers are shared on purpose, so the unresolved
cross-release participant question does not affect this split. Using normal reading
only also avoids the task/topic overlap described in the paper review.

1. Merge sentence keys with identical alphanumeric tokens or token Jaccard of at
   least 0.7. This merges 4 pairs, giving 590 groups.
2. Put every group that occurs in both releases in training (53 groups).
3. Within each release and word-count tertile, shuffle with seed 42 and assign 15%
   to test, 15% to validation and the rest to training.

| Split | Observations | Sentences | ZuCo1 sentences | ZuCo2 sentences |
|---|---:|---:|---:|---:|
| Train | 6,894 | 432 (50 in both releases) | 226 | 256 |
| Validation | 1,164 | 81 | 37 | 44 |
| Test | 1,175 | 81 | 37 | 44 |

These sizes come from the sentence text only, with no EEG read. All 30 readers
appear in every split. Training has about three times the 2,245 observations used
in the later pilots.

## Models and arms

No new architecture, target or preprocessing is introduced. Whitening, attribute
heads, extra reading tasks and text generation are excluded so one change is tested
at a time.

| Arm | Input | Why it is here |
|---|---|---|
| Ridge, real | Existing statistical EEG features | Fast baseline |
| Ridge, noise | Matched Gaussian noise | Does the score need EEG? |
| Ridge, shuffled | Real EEG, shuffled training sentence classes | Does the pairing matter? |
| CNN, real | Existing waveform encoder | Learned model |
| CNN, noise | Matched Gaussian noise | Same check for the CNN |
| CNN, shuffled | Real EEG, shuffled training sentence classes | Same check for the CNN |
| Word counts and timing only | Duration and word count, no amplitudes | Is structure enough? |

- **Fitting:** preprocessing fits on training observations only.
- **Ridge:** selects one of seven alphas on the validation primary metric.
- **CNN:** batch size 8, at most 50 epochs, patience 8, at most 12 hours per arm. It
  selects its checkpoint on validation alignment loss.
- **Known limit:** the matched noise keeps each observation's per-channel amplitude,
  so it is a limited null model.

## Measurement

**Primary score:** expected 10-way accuracy, with chance at 10%. The correct sentence
competes with random distractors from the validation or test sentences of the same
release. Restricting to one release stops a model scoring well by recognizing the
release. Scores average within each reader, then equally across 30 readers.

**Also reported, not used in the gate:**

- Expected 2-way, 4-way and 24-way accuracy, the paper's values of N.
- Expected 10-way accuracy among the 20 candidates closest in word count.
- Full-pool Recall@1/5/10 and MRR. MRR is not an accuracy.

## Gate

Applied on validation, separately for ridge and CNN. A model passes only if all
three hold:

1. Real EEG beats noise, shuffled and word-count/timing controls by at least
   2 percentage points on the primary score.
2. Real EEG beats noise for at least 20 of 30 readers.
3. The 95% bootstrap lower bound of real minus noise is above zero, both when
   resampling readers and when resampling sentences (10,000 resamples each).

This is a stopping rule set in advance. It is not proof of semantic decoding.

## What happens next

| Outcome | Action | What may be claimed |
|---|---|---|
| A model passes | Score that model and its controls once on test sentences | The test result, for familiar readers and new sentences only |
| Neither passes | Stop and never score test sentences | The easier split did not establish an EEG gain with these small models |
| Run fails or hits the time limit | Report it and ask the owner | Nothing about performance |

Stop after this run in every case. Extra seeds, settings or architectures need a new
plan. Whitening would be the next candidate only after a pass.

These recordings and sentences have all been used in earlier project work, so even
the new test sentences are not untouched data. No model here trains on them, but a
pass is exploratory and not independent confirmation.

## Work needed before the run

Write these first and test them on synthetic fixtures only:

1. A `sentence_holdout` split with its leakage checks.
2. Same-release candidate pools and the word-count-matched 10-way score.
3. The word-count/timing control.
4. A runner that refuses an existing output directory, checks the text cache and
   signal hashes, and hashes protected prior artifacts before and after.

Disk check: C: had about 14 GB free on 2026-11-05. Checkpoints for these small models
should use little of that, but check free space again before starting. Outputs go to
`results/sentence_holdout_pilot_v1_20261105/`, and existing datasets, targets,
checkpoints and results stay unchanged.
