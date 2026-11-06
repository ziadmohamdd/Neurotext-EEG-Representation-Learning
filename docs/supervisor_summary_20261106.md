# ZuCo retrieval audit: summary for supervisor

6 November 2026, Ziad Mohamed

I have not improved EEG decoding: on held-out list blocks, EEG never beat matched noise. I may still have a paper in how ZuCo retrieval is scored, but each part has prior work, so only the combination looks new. All of this is exploratory, and I would like your view before going further.

## Evidence

Scores are expected 10-way accuracy unless marked, with chance at 10%. No final test data was scored. Word-locked scores use centred scoring, so they are not on exactly the same scale as the others.

| Result | Key numbers | Data used |
|---|---|---|
| GLIM's score drops within a task | 21.46% mixed (21.91% reported in the SemKey paper), 9.73–14.25% within a task | GLIM's public outputs, 156 sentences |
| The gain comes from telling corpora apart | 2-way: movie reviews vs biographies 87.36–93.57%, biography tasks vs each other 51.89–60.66% | Same outputs |
| Random splits are a weak test | Neighbouring text with true position, no EEG: 51.09% mean over 20 random splits, against 21.10% for our v1 CNN on its own split | Our text data; validation data for the CNN |
| EEG reveals list position | Predicted vs true position correlates at 0.80 (ZuCo 1.0) and 0.88 (ZuCo 2.0) | 53 validation sentences, familiar readers |
| Whole-sentence EEG fails on held-out blocks | CNN 10.02% (noise 9.72%). Averaged across readers 11.25% (noise 12.13%, timing 12.07%) | 100 and 255 reused sentences |
| Word-locked EEG fails on held-out blocks | All readers 14.55% (noise 16.25%). One reader 11.72% (noise 11.96%). With position, word count and reading time removed: 10.02% | 255 reused sentences, 83,798 word rows |
| Gaze beats word-locked EEG | Gaze and word properties with no EEG: 26.05% | Same sentences |
| New readers show no gain | Whole-sentence 9.67% (noise 11.23%). Word-locked 13.22% (noise 13.31%) | Same sentences, 15 fit and 15 scored readers |
| Same sentence across releases: no match beyond noise | Real 28.18% (noise 26.68%), list position alone 54.39%, word count and reading time 52.74%. With these removed, 18.40% (noise 16.06%) | 48 sentences read in both releases, nothing fitted |

## What this does not show

- **No meaning in the EEG:** these tests used summary features and linear models, and the main word input starts at 4 Hz. [Raghavan et al. 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205611/) found fixation-locked EEG on ZuCo tracks word features. So the claim is only no above-noise retrieval under this split.
- **GLIM's own metric:** I tested generated text only. GLIM's within-group retrieval (8.15% against 4.17% chance) does not mix tasks. It also says nothing about neighbours: GLIM's outputs do not follow them.
- **Other models:** SemKey is untested because its outputs are not public.
- **Cause:** real EEG is needed for the movie-review gap, as one pair scores 89.94% against 48.53% on noise input. The EEG could reflect content or recording blocks. The same-sentence test does not settle this, as it covers biography sentences only, and the shared sentences also sit in similar list positions (correlation 0.70).
- **Side results:** first-fixation waveform scored 12.81% against noise 9.72%, and time-binned same-sentence matching 12.75% against 5.70%. Neither has an interval or gaze control, and both could be eye-movement traces.
- **Size:** 255 sentences across 13 blocks, all used in earlier work, so any result needs confirming.

## Proposed paper

The claim would be that mixed-task scores and random splits on ZuCo can produce above-chance retrieval that fair held-out tests do not support. My [systematic search](literature_search_20261106/REVIEW.md) found each of the four claims partly published, and none published as a whole:

- **Closest paper:** [Zhang et al., ACL 2026](https://aclanthology.org/2026.acl-long.61/) says ZuCo scores match noise, but does not test list position.
- **Blocks:** [Hollenstein et al. 2021](https://ar5iv.labs.arxiv.org/html/2112.06310) already shows EEG classifying ZuCo 2.0 recording blocks.
- **Public code:** [bleepbloop301](https://github.com/bleepbloop301/eeg2text-reproductions), [RahmaDawy](https://github.com/RahmaDawy/eeg-language-decoding) and [thestonedape](https://github.com/thestonedape/task-aware-eeg2text) overlap heavily and are not peer reviewed. One mentions a working manuscript.
- **Against us:** [McGuire and Moshfeghi, SIGIR 2025](https://arxiv.org/html/2412.06695v3) report 12.39% against 4.13% noise at P@5 on ZuCo, on a different task and split. [Marquardt et al.](https://arxiv.org/abs/2608.20186) report a strict positive on their own 49-hour single-reader data.

Still needed:

1. A test on a second published model, using SemKey's outputs or a retrained copy. Retraining needs ZuCo task-specific reading data, which I do not have locally.
2. A cleaner test of content against recording block. The same-sentence test was a first attempt, and repeats read by the same people in different tasks would be stronger, but need more data.
3. A confirmation on data not yet used, which is the v2 test blocks.

None of this raises decoding accuracy. The search was not exhaustive: keyword rules removed many records before reading, and Google Scholar was not searched.

## Questions for you

1. Is a benchmark audit an acceptable outcome, given that decoding has not improved?
2. Is the paper worth pursuing, given the public code overlap, and should I contact those authors?
3. I asked SemKey's authors for their outputs on 6 November. If they cannot share them, should I retrain the model myself?
4. Should I stop decoding work, or test the first-fixation waveform once more under a frozen plan?

Full results: [GLIM output check](research_results/glim_output_check_20261106/REPORT.md), [averaged-EEG test v5](research_results/averaged_eeg_v5_20261106/REPORT.md), [word-locked test v6](research_results/word_locked_v6_20261106/REPORT.md), [same-sentence test v7](research_results/same_sentence_v7_20261106/REPORT.md), [position test v4](research_results/position_bridge_v4_20261105/REPORT.md).
