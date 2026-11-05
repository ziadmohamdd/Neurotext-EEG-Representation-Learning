# Block-holdout pilot: plan v2

**Status: approved by the project owner on 2026-11-05.** It runs only after
[sentence-holdout pilot v1](sentence_holdout_pilot_plan_v1_20261105.md) has finished.
SHA-256 of [the config](../configs/block_holdout_pilot_v2_20261105.json):
`eef247790d718091b5f3e5357feb1989ceb7a3881f852868170ad89a641f9006`. Both files are frozen. Any later change needs a new version.

## Why v2 exists

An audit during pilot v1 found problems that v1 cannot fix:

| Finding | Measured on v1 validation data | v2 response |
|---|---|---|
| Random sentence holdout leaks list order | EEG features predict list position at r = 0.79 (ZuCo1) and 0.71 (ZuCo2). Text from the two nearest training sentences reaches 54% at 10-way with no EEG | Hold out blocks, add buffers, add a position control |
| Noise control keeps session drift | Noise features still predict position at r = 0.67 and 0.60 | Keep it as a conservative control and add a within-block score |
| ZuCo2 keeps 50 Hz mains noise | 50 Hz power is 6.56 times 40-45 Hz power in ZuCo2, about 0 in ZuCo1 | Notch filter at load time for both releases |
| Long sentences are cut | 31% of ZuCo1 and 21% of ZuCo2 sentences exceed 8 seconds | Keep up to 30 seconds in pipeline F |
| Reader scale differs | RMS varies about 2.6 times between readers | Per-reader normalization in pipeline F |
| Cz is flat | Flat in all 300 sampled sentences | Drop it |

List position is taken from sentence IDs. That it matches presentation order is
inferred from the strong drift result and was not checked against stimulus logs.

## Question and hypothesis

With familiar readers and new sentence blocks kept apart from training in list order,
does real EEG beat noise, shuffled pairs, word-count/timing and list-position controls?

**Hypothesis:** real EEG reaches at least 2 percentage points higher expected 10-way
accuracy than each control. The hypothesis fails if any gate condition fails on
validation. A drop below v1's 16.44% would be an acceptable and informative outcome,
since v2 is meant to remove an inflating factor.

## Place in the project brief

This informs brief steps 2 (preprocessing), 4 (noise control and sentence information)
and 7 (limitations). It does **not** replace step 6: unseen-subject testing remains the
required generalization result. Results must be labelled "familiar readers, new
sentence blocks".

## Cohort and split

Existing ZuCo 1.0 and 2.0 normal-reading observations after duration QC: 9,233
observations from 30 readers. No download is needed.

In each release, the list gets four blocks of 25 sentences: two for validation and two
for test. Each block has 10 buffer sentences on both sides that are never used, and
neighbouring buffers may overlap. Lag 10
is where text similarity between list neighbours falls close to its overall average.

| Set | Observations | ZuCo1 sentences | ZuCo2 sentences |
|---|---:|---:|---:|
| Train | 4,487 | 124 | 183 |
| Buffer (unused) | 1,933 | Not scored | Not scored |
| Validation | 1,427 | 50 | 50 |
| Test | 1,386 | 50 | 50 |

These exact sizes come from sentence text only, with no EEG read. Sentences that repeat, have a near-duplicate or occur in both releases never enter held-out blocks. Buffers shrink training from v1's 6,894
observations, which is the cost of a cleaner test. It remains well above the 2,245
observations in the later ZuCo1 pilots.

## Pipelines and arms

| Pipeline | Filter | Channels | Window | Normalization |
|---|---|---|---|---|
| V | 50 Hz notch | Without Cz | 8 seconds | Shared across readers |
| F | 50 Hz notch | Without Cz | Up to 30 seconds | Per reader |

| Model | Pipelines | Arms |
|---|---|---|
| Ridge | V and F | Real, noise, shuffled pairs |
| CNN (seeds 42, 43, 44) | F | Real, noise, shuffled pairs |
| Word counts and timing only | None | One arm, no amplitudes |
| List position only | None | One arm, no amplitudes |

- **Fixed settings:** these match v1, apart from the three CNN seeds. CNN arms in v1
  ran about one minute per epoch, so three seeds should be affordable. Longer windows
  will slow that down.
- **Bundled changes:** F changes the window and normalization together. A pass or fail
  cannot separate them, and doing so would need a later plan.
- **Left out:** whitening, extra reading tasks, extra prediction targets, new encoders
  and text generation, so that the test is settled first.

## Measurement

**Primary score:** expected 10-way accuracy against all 50 sentences of the same split
and release, averaged within each reader and then across 30 readers. Chance is 10%.

**Within-block score:** expected 10-way accuracy against only the 25 sentences of the
query's own block. Those sentences sit close together in the session, so list position
gives little help here. This is the strictest check against order leakage.

Also reported, not used in the gate: 2-way, 4-way and 24-way accuracy; the
word-count-matched 10-way score; Recall@1/5/10; MRR, which is not an accuracy; and how
well EEG predicts list position under the new split.

## Gate

Applied on validation, separately to ridge V, ridge F and CNN F. A model passes only
if all four hold:

1. Real EEG beats noise, shuffled pairs, word counts/timing and list position by at
   least 2 percentage points on the primary score.
2. Real EEG beats noise by at least 2 percentage points on the within-block score.
3. Real EEG beats noise for at least 20 of 30 readers.
4. The 98.33% bootstrap lower bound of real minus noise is above zero, both when
   resampling readers and when resampling sentences (10,000 resamples each). The
   stricter level allows for testing three models.

With only four validation blocks, these intervals cannot capture block-to-block
variation, so they may be too narrow. This is a stopping rule set in advance. It is not
proof of semantic decoding.

## What happens next

| Outcome | Action | What may be claimed |
|---|---|---|
| A model passes | Score that model and its controls once on test blocks | The test result, for familiar readers and new blocks only |
| None passes | Stop and never score test blocks | With order leakage reduced, these small models did not establish an EEG gain |
| Run fails or hits the time limit | Report it and ask the owner | Nothing about performance |

Stop after this run in every case. A pass would make whitening, then a time-aware
encoder and extra targets, the next candidates under new plans. A fail would favor
writing up the result over further tuning on this data.

## Limits to keep visible

- v1 validation scores shaped this design, and v1 may score its own test sentences.
  Some v2 held-out sentences overlap those sets, so v2 is exploratory and not an
  independent confirmation.
- Session drift still exists inside the data. Blocks and buffers reduce how much it
  can help, but cannot remove it entirely.
- Normal reading only, so this says nothing about the other ZuCo tasks.
- The readers are shared, so nothing here shows generalization to new people.

## Leakage safeguards

The runner writes `leakage_audit.json` and refuses to start if any check fails:

- No sentence, observation or signal file appears in two splits.
- No held-out sentence has wording close to any sentence outside its split (token
  Jaccard below 0.7, token containment below 0.8), including buffer sentences.
- No held-out sentence embedding nearly matches a training target (cosine below 0.9).
- Training stays more than 10 list positions from every held-out sentence.
- Every held-out reader has training statistics.

Tests also confirm that held-out signals cannot change fitted statistics, and that a
failed gate never opens test signals. Normalization and model weights fit on training
rows only, and validation only selects settings.

These checks cannot rule out everything. Shared readers are by design, session drift
remains in the signals, and upstream ZuCo session preprocessing cannot be audited here.

## Work needed before the run

Write these first and test them on synthetic fixtures only:

1. A block split with buffer, near-duplicate and shared-sentence checks.
2. Load-time notch filtering, Cz removal, the 30-second window and per-reader
   normalization. The prepared signal files stay unchanged.
3. The list-position control, within-block scoring and three-seed CNN aggregation.
4. A runner that refuses an existing output directory, checks the approved config hash
   and hashes protected prior artifacts before and after.

Check GPU memory on the longest sentences (up to 30 seconds after cropping) with
synthetic arrays of that size first. C: had about 14 GB free on 2026-11-05. Outputs go
to `results/block_holdout_pilot_v2_20261105/`, and existing datasets, targets,
checkpoints and results stay unchanged.
