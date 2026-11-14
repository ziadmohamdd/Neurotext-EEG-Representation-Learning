# Plan for the audit paper's weaknesses

6 November 2026. **Update:** [same-sentence test v7](research_results/same_sentence_v7_20261106/REPORT.md) has now run and found no match beyond noise. Section 1 below is the original proposal, and its scoring changed before freezing. No data was downloaded. Counts come from local manifests, OSF file listings and SemKey's public repository.

| Weakness | Fix | Needs | Status |
|---|---|---|---|
| Content or recording block | Same-sentence test, v7 | Local data only | Run, no match beyond noise; only partly resolved |
| Only one published model | Ask SemKey's authors for outputs, then retrain only if refused | An email from the owner, or 52.1 GB and a larger GPU | Partly addressed: [v8](research_results/public_outputs_check_v8_20261112/REPORT.md) checked two more models. SemKey still waits on the authors, emailed 2026-11-06 |
| Reused sentences | Score the v2 test blocks once, last | The owner's decision and a final design | Keep closed for now |
| Others may publish first | Finish the local test, then decide with the supervisor | Supervisor's answer | Waiting |

**Update, 12 November:** [a reading-only check](assumption_and_outputs_check_20261112.md) found public outputs for two more models (Wang and Ji's BART model and Jo et al.'s T5 version), and both dataset papers confirm a fixed presentation order. [Check v8](research_results/public_outputs_check_v8_20261112/REPORT.md) then ran on those outputs: neither model retrieves its sentence above chance and noise, so three published models are now audited, but only GLIM shows the mixed-task pattern.

**Update, 14 November:** [first-fixation test v9](research_results/first_fixation_v9_20261114/REPORT.md) retested the last decoding lead, and it did not hold: the waveform added 0.10 points to gaze. Decoding work has stopped. The project owner decided to finish without waiting for the supervisor or for SemKey's authors, so SemKey stays untested and is listed as a limitation. [Paper draft 2](paper_draft_20261114.md) adds two analyses from saved results only: [effect limits](research_results/paper_analyses_20261114/effect_limits.json), the largest EEG gain each test still allows (at most 2.37 points over gaze), and an [exact corpus-only ceiling](research_results/paper_analyses_20261114/label_ceiling.json), which scores 19.53% at 10-way against GLIM's 21.46% on the same test set. That ceiling applies to any model scored on that test composition, which partly answers the single-model weakness.

## 1. Content or recording block (v7)

**Idea:** 51 sentence texts appear in both ZuCo 1.0 and ZuCo 2.0 normal reading, so the same words were read in different recording blocks by different people. Of these, 48 are usable in both releases; the other 3 touch the v2 test zone and stay closed.

**Question:** is a sentence's averaged EEG closer to the same sentence in the other release than to other sentences?

- **Content predicts:** same-sentence matches across releases beat other-sentence matches.
- **Block predicts:** sentences from the same block look alike within a release, with no same-sentence match across releases.

**Why it suits this project:**

- These 48 sentences were excluded from scoring in v5 and v6, because they repeat across releases. They were never scored as held-out sentences there, though they were available for fitting.
- The planned test fits no decoder, so there is nothing to train on neighbours.
- About 11 ZuCo 1.0 and 17 ZuCo 2.0 readers contribute per sentence on average.

**Proposed primary score:** expected 48-way accuracy across releases, chance 2.08%, using v5's features averaged within each release and standardised within each release.

**Proposed controls:** shuffled sentence pairing, matched noise, and a partial check that removes word count, reading time and position. Word count is the main risk, because long sentences produce similar EEG in both releases whatever they mean.

**Limits:**

- **Small:** 48 sentences give a wide interval.
- **A null is weak:** the two releases differ in readers and in recording setup, so failing to match does not show that content is absent.
- **A positive is strong:** only if it survives the partial check.
- **It does not settle the movie-review gap:** in ZuCo, movie reviews were read only in the sentiment task, so that content is tied to its own recording session. No sentence is shared between that task and normal reading in our data. This test can only address biography sentences.

**Possible stronger version:** the same readers saw some sentences in both normal and task-specific reading. That removes the change of reader, but needs task-specific data. ZuCo 1.0 alone is 8.375 GB. The number of repeated sentences has not been counted, which would take only the small `task_materials` files.

## 2. A second published model (SemKey)

**Cheapest route: ask the authors.** SemKey's repository has no releases and no generated outputs. If the authors share their generated test text, the [GLIM check](research_results/glim_output_check_20261106/REPORT.md) can be repeated directly on it, with no training.

**Retraining, only if that fails:**

| Files needed (Matlab folders) | Size |
|---|---:|
| ZuCo 1.0 `task1- SR` | 11.002 GB |
| ZuCo 1.0 `task3 - TSR` | 8.375 GB |
| ZuCo 2.0 `task2 - TSR` | 32.745 GB |
| **Total** | **52.122 GB** |

These sizes are for the main OSF folders, which match the size of the normal-reading files held locally. OSF also lists a January 2019 ZuCo 1.0 version at about twice the size (22.033 and 17.437 GB, 72.215 GB in total). SemKey's README does not say which version it used.

**Why retraining is hard here:**

- **Disk:** C: has about 6.8 GB free and D: about 67 GB. The 52.122 GB fits on D:, but preprocessing and model files also need space, and the January 2019 version would not fit.
- **GPU:** this laptop has an RTX 3050 with 4 GB. SemKey's scripts use `flan-t5-large` with a batch size of 24 at the end-to-end stage, so it probably needs a larger GPU. This has not been tried.
- **Fidelity:** a retrained copy may not reach the published scores, and the paper's and code's split settings disagree. A low score would then prove little.

**Useful detail for the paper:** SemKey's training scripts define topic labels of "Biographies and Factual Knowledge" and "Movie Reviews and Sentiment". This fits the corpus claim, but how much those labels drive its score is untested.

## 3. Confirmation on unused data

The v2 test blocks are the only clean data left and can be scored only once, so they come last. Scoring needs a new frozen plan, the final model and controls, and the owner's decision.

## Recommended order

1. Freeze and run v7, which needs no download. Done.
2. Send the supervisor summary with the v7 result if it is ready.
3. Ask SemKey's authors for their outputs. Sent on 2026-11-06.
4. Decide on retraining only if they cannot share them.
5. Score the v2 test blocks last.
