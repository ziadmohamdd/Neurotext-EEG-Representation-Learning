# ZuCo retrieval audit: summary for supervisor

6 November 2026, Ziad Mohamed

I have not improved EEG decoding, but I may have a paper in how ZuCo retrieval is scored. On GLIM's public outputs, 10-way accuracy falls from 21.46% across mixed tasks to 9.73–14.25% within a single task (chance is 10%). This analysis was exploratory, so I would like your view before going further.

## Evidence

Scores are expected N-way accuracy. Chance is 50% at 2-way and 10% at 10-way. No final test data was scored.

| Result | Key numbers | Data used |
|---|---|---|
| GLIM's score drops within a task | 10-way: 21.46% mixed (21.91% reported in the SemKey paper), 9.73–14.25% within a task | GLIM's public outputs, 156 sentences |
| The gain comes from telling corpora apart | 2-way: movie reviews vs biographies 87.36–93.57%, biography tasks vs each other 51.89–60.66%. A task label with no EEG scores 68.06%, GLIM 68.29% | Same outputs |
| Random splits are a weak test | 10-way with neighbouring text and true position, no EEG: 51.09% mean over 20 random splits. On GLIM's own sentences: 52.76%, against GLIM's 14.84% | Our text data; 36 GLIM sentences |
| EEG reveals list position | Predicted vs true position correlates at 0.80 (ZuCo 1.0) and 0.88 (ZuCo 2.0). 10-way from position alone: 15.41%, shuffled 9.70%, our v1 CNN 21.10% | Our validation data, 53 sentences, familiar readers |
| Our CNN fails when blocks are held out | 10-way: 10.02% | Our validation data |

## What this does not show

- **Neighbour route in GLIM:** GLIM's outputs do not follow list neighbours. Their similarity stays flat from near to far (0.164 to 0.165), while true sentences fall from 0.249 to 0.119.
- **GLIM's own metric:** I tested generated text only. GLIM's within-group retrieval (8.15% against 4.17% chance) does not mix tasks.
- **Other models:** SemKey is untested because its outputs are not public.
- **Cause:** real EEG is needed, as one movie-review pair scores 89.94% against 48.53% on noise input. The EEG could reflect content or recording blocks, and these files cannot separate them.
- **Sample size:** within-task pools hold only 15–40 sentences.
- **Novelty:** my literature search was not systematic, and citations were not traced. The closest work I found is [Hollenstein et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/) and [Zhang et al., ACL 2026](https://aclanthology.org/2026.acl-long.61/). I have not yet read every paper in full.

## Proposed paper

The claim would be that mixed-task N-way scores on ZuCo mostly measure telling corpora apart, not decoding sentences. Three things are still needed:

1. A systematic literature search with citation tracing, to check this is new.
2. A test on a second model, using SemKey's outputs or a retrained copy. Retraining needs ZuCo task-specific reading data, which I do not have locally.
3. A test that separates content from recording block.

None of this raises decoding accuracy, and the new-reader goal in the project brief is still unsolved.

## Questions for you

1. Is a benchmark audit an acceptable outcome, given that decoding has not improved?
2. Should I ask SemKey's authors for their outputs, or retrain the model myself?
3. Should the new-reader work come before the audit?

Full results: [GLIM output check](https://github.com/ziadmohamdd/Neurotext-EEG-Representation-Learning/blob/feature/block-holdout-pilot/docs/research_results/glim_output_check_20261106/REPORT.md), [position test v4](https://github.com/ziadmohamdd/Neurotext-EEG-Representation-Learning/blob/feature/block-holdout-pilot/docs/research_results/position_bridge_v4_20261105/REPORT.md), [published splits check](https://github.com/ziadmohamdd/Neurotext-EEG-Representation-Learning/blob/feature/block-holdout-pilot/docs/published_splits_check_20261105.md).
