# Same-sentence test: plan v7

**Status: approved by the project owner on 2026-11-06.** [The config](../configs/same_sentence_v7_20261106.json) is frozen, and its SHA-256 is recorded in `scripts/run_same_sentence.py`. Any later change needs a new version.

## Question

48 usable sentences appear in both ZuCo 1.0 and ZuCo 2.0 normal reading, so the same words were read by different people in separately recorded lists. Is a sentence's averaged EEG closer to its copy in the other release than to other sentences, beyond noise, length, reading time and list position?

- **A clear match** would be content in the EEG that does not depend on recording block.
- **No match** is weak evidence, because readers and recording setup both differ between releases.

The test covers biography sentences only. It cannot settle the movie-review gap, as movie reviews were not in the normal-reading data used here.

## Design

1. **Input:** the stored v5 features, read-only. No EEG file is opened and nothing is fitted to meaning.
2. **Scaling:** each reader is z-scored from rows outside the 48 sentences. Features are averaged across readers per sentence, then z-scored across the 48 within each release.
3. **Score:** expected 10-way across releases by cosine, mean of both directions, with chance at 10%.
4. **Controls:** matched noise, re-paired sentences, word count and duration alone, list position alone, and a partial check that removes position, its square, word count and duration from real and noise.
5. **Block effect, secondary:** within each release, whether sentences within 25 list positions look more alike than distant ones, using all usable sentences.

## Decision

Each control needs a gap of at least 2 points, and every 95% lower bound above 0. Sentences are resampled for every control, and readers within release for noise, re-pairing and the partial check.

| Result | Label |
|---|---|
| Beats noise and re-pairing, and clears timing, position and partial | The same sentence matches across releases beyond the controls |
| Beats noise and re-pairing, but not the others | The match does not exceed word count, reading time or position |
| Does not beat noise and re-pairing | No same-sentence match found with these features |

## Protected data and leakage

- v1 test sentences, v2 test blocks and their buffers stay closed. A text with any protected copy is left out, which removes 3 of the 51 shared texts.
- None of the 48 sentences was scored as held out in v5 or v6, because repeats were excluded from scoring. They were available for fitting there, but this test fits nothing.
- The run stops if the stored features' hash or shape, the channel lists, or the number of pairs differ from the plan.

## Synthetic checks before freezing

Over 20 seeds each, no signal, word-count-only, duration-only and position-only signals earned the first reading 0 times. A planted same-sentence signal earned it on 0 of 10 at strength 0.05 and 0.10, where the real score was 12.1% and 22.5%, and 10 of 10 from strength 0.15, where it was 44.4%. **So on these fixtures, a match below roughly 25% could be missed.** A real signal may differ from the fixtures.

## Changes made after approval and before freezing

No score had been computed when these were made.

- **Primary score:** the [draft](paper_weaknesses_plan_20261106.md) proposed 48-way, with chance at 2.08%. This uses expected 10-way, which matches the rest of the project and depends less on a single top rank. 48-way stays as a secondary score.
- **Block intervals:** none. The 48 sentences are scattered across the lists, so they do not form blocks. Readers are resampled instead.
- **Bugs found in testing:** the block-effect control first rotated the list, which keeps neighbours together and could not detect drift. It now shuffles the list. Reader resampling used the wrong index for the second release, which the tests had missed; it is fixed, with a test.
- **Stricter than earlier versions:** the partial check is also resampled by reader.

## Stop rule

Stop after this run, whatever the outcome. No extra settings or features under this plan. No final test data is scored.
