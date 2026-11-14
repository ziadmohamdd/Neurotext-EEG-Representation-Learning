# What ZuCo EEG-to-text retrieval scores measure: an audit of task mixing, list position and controls

**Draft 1, 14 November 2026.** Ziad Mohamed. Not for submission. Every number below comes from a report in `docs/research_results/`, and each section links its source. Items marked **TODO** need work before submission. References marked **check** have not been verified against the paper itself.

## Abstract

EEG-to-text models trained on the ZuCo reading corpora are often scored by N-way retrieval: whether a model's output is closer to its own sentence than to N−1 others. We audit what this score measures. First, on the public test outputs of GLIM, a recent model, the mixed-task 10-way score of 21.46% (chance 10%) falls to 9.73–14.25% within any single reading task, and its 2-way gain comes almost entirely from separating movie reviews from biographies (87.36–93.57%), not biography tasks from each other (51.89–60.66%). Second, random sentence splits are a weak test: because all readers saw the sentences in the same fixed order, the text of a held-out sentence's list neighbours, with no EEG, scores 51.09% at 10-way on average over 20 random splits, and EEG predicts a held-out sentence's list position with a correlation of 0.80–0.88. Third, under frozen held-out block splits with buffers, EEG did not beat matched noise with whole-sentence features (11.25% against 12.13%), word-locked features (14.55% against 16.25%), or first-fixation waveforms (12.81% against 9.72%, gap not clear). Gaze and word properties with no EEG scored 26.05%, and adding the first-fixation waveform to them added 0.10 points. The public outputs of two further models show no retrieval above chance and noise. We do not claim that ZuCo EEG holds no language information. We claim that the common scores, as usually computed, do not show it, and we propose a set of controls that would.

**TODO:** shorten to 200 words for most venues.

## 1. Introduction

Decoding language from non-invasive brain recordings would matter for communication aids and for the science of reading. ZuCo 1.0 and 2.0 (Hollenstein et al. 2018, 2020) are the standard public datasets for EEG recorded during natural reading, with eye tracking that marks each fixated word. A line of work trains models from ZuCo EEG to text, and scores them with generation metrics such as BLEU and ROUGE, or with N-way sentence retrieval.

Several problems with this line of work are known. Models trained with teacher forcing can produce good-looking text from noise input (Jo et al. 2024, **check**), and recent work reports that ZuCo scores match noise baselines (Zhang et al., ACL 2026). We extend this critique to the retrieval scores that newer models report, and ask three questions:

1. **Task mixing:** when test sentences come from different reading tasks and corpora, how much of an N-way score comes from telling the corpora apart?
2. **Split design:** do random sentence splits let list position or recording drift stand in for content?
3. **Controls:** under a split that removes these routes, does EEG beat matched noise and a gaze-only predictor?

Our contributions:

- A breakdown of a published model's own public outputs by task and by distractor type (Section 4).
- A no-EEG neighbouring-text baseline that shows how easy random ZuCo splits are, and evidence that EEG encodes list position (Section 5).
- Frozen, pre-registered tests on held-out list blocks with buffers, using matched noise, shuffled pairs, list position, timing and gaze controls, for whole-sentence, reader-averaged, word-locked and first-fixation EEG (Section 6).
- A check of two further published models' public outputs, and a cross-release same-sentence test (Sections 4.3 and 6.4).

Each idea has partial precedent (Section 2). We believe the combination, run on one data set under frozen plans, is new.

## 2. Related work

**ZuCo and its benchmark.** ZuCo 1.0 has 12 readers and ZuCo 2.0 has 18 different readers (Hollenstein et al. 2021, which states the datasets used separate participants). Both present sentences in a fixed order that was identical for all readers. Hollenstein et al. (2021) showed that EEG can classify ZuCo 2.0 recording blocks, with errors mostly in neighbouring blocks, and the ZuCo benchmark (Hollenstein et al. 2023) shuffled its test sentences to stop models identifying a block.

**EEG-to-text on ZuCo.** Wang and Ji (2022) trained BART from word-level EEG features. Jo et al. (2024, **check**) showed that such models produce similar text from noise input. GLIM (**check** citation) and SemKey (**check** citation) report N-way retrieval scores; SemKey's paper reports GLIM at 21.91% for 10-way.

**Critiques.** Zhang et al. (ACL 2026) report that ZuCo decoding scores match noise and attribute this to EEG instability, but do not test list position. McGuire and Moshfeghi (SIGIR 2025) report above-noise retrieval at P@5 (12.39 against 4.13) on a different task and split, and Marquardt et al. (2026, arXiv 2608.20186) report a strict positive on their own single-reader data. Raghavan et al. (2025) find that fixation-locked ZuCo EEG tracks word features.

**Unreviewed public code.** Three public repositories overlap with our claims: bleepbloop301/eeg2text-reproductions (task and position controls for GLIM-style scores, including a keyword rule that recovers the corpus from GLIM's released outputs 97% of the time), RahmaDawy/eeg-language-decoding (noise and gaze-style baselines at the sentence level) and thestonedape/task-aware-eeg2text (task margin). We found no paper from them. We cite them as prior work.

**Leakage in EEG decoding.** Li et al. (2020, *IEEE TPAMI*) showed that block designs let EEG classifiers exploit temporal correlation rather than stimulus content. **TODO:** add the replies to that paper and other autocorrelation work from the search records.

Our search method and counts are in `docs/literature_search_20261106/`. It was not exhaustive: keyword rules removed records before reading, and Google Scholar was not searched.

## 3. Data and scoring

**Data.** We use the normal-reading tasks of ZuCo 1.0 and 2.0 with the supplied preprocessed EEG (105 channels, Cz removed for most analyses) and word-level features. Readers: 12 in ZuCo 1.0 and 18 in ZuCo 2.0, 30 in total. Sentences from our earlier v1 test set and a reserved v2 test zone were never fitted or scored in any analysis here (Section 7).

**Expected N-way accuracy.** For a query sentence, we compute the exact expected top-1 accuracy over all ways of drawing N−1 distractors from a pool, with ties counted as misses. Chance is 1/N. Unless stated otherwise, scores are 10-way, one equal-weight score per sentence. Text targets are all-MiniLM-L6-v2 sentence embeddings.

**Pools.** *Mixed* pools draw distractors from all tasks. *Within-task* pools draw them from the query's own task. *Within-block* pools (Section 6) draw them from the query's own held-out list block, which makes coarse list position much less useful.

**Decision rule.** In frozen tests, a comparison is *clear* only if the gap is at least 2 points and every available 95% lower bound is above 0. Bounds come from resampling sentences, blocks and, where defined, readers.

## 4. Task mixing in published outputs

### 4.1 GLIM's score comes from telling corpora apart

We scored GLIM's three public output files (repository commit `e1f202c`), with no EEG read: 1,889 rows and 156 sentences after protected sentences were removed. [Source](research_results/glim_output_check_20261106/REPORT.md).

| Predictor | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| GLIM, as reported in the SemKey paper | 66.83 | 42.45 | 21.91 | 10.74 |
| GLIM outputs, our scoring | 67.26 | 42.71 | 21.46 | 10.47 |
| GLIM with noise input | 51.55 | 26.50 | 11.26 | 5.06 |

Our scoring reproduces the published numbers closely. Within a single task, the 10-way score is 9.73% to 14.25% (Table 2). At 2-way, GLIM separates movie reviews from any biography task 87.36–93.57% of the time, but biography tasks from each other only 51.89–60.66%. Same-task pairs score 51.75% (interval 50.27 to 53.30), and different-task pairs 72.83%.

**Table 2.** 10-way, one row per sentence per trial.

| Pool | Sentences | GLIM | Noise input |
|---|---:|---:|---:|
| All tasks | 156 | 21.71 | 11.10 |
| ZuCo 1.0 movie reviews | 40 | 10.53 | 9.44 |
| ZuCo 1.0 normal reading | 15 | 14.13 | 11.50 |
| ZuCo 1.0 task-specific reading | 39 | 9.73 | 12.31 |
| ZuCo 2.0 normal reading | 24 | 14.25 | 11.58 |
| ZuCo 2.0 task-specific reading | 38 | 11.30 | 10.06 |

A predictor that knows only the task label, and outputs the mean text of the other test sentences in that task, scores 68.06% at 2-way, matching GLIM's 68.29%.

### 4.2 What drives the corpus gap

The gap is not the task prompt alone: two ZuCo 1.0 tasks share the same prompts in GLIM's code, yet that pair scores 89.94%. Real EEG is needed, as noise input scores 48.53% on the same pair. Movie reviews were read only in their own recording sessions, so the EEG could reflect content or recording block. These outputs cannot separate the two.

### 4.3 Two further models

Wang and Ji's BART model and Jo et al.'s T5 variant have public test outputs (79 usable sentences). At 10-way across movie reviews and biographies, BART scored 10.01% (noise input 10.09%), and its outputs contain only 4 distinct texts. T5 scored 12.38% (noise 9.61%), with a gap over chance of 2.38 points (interval −0.65 to 5.63). A predictor that knows only the corpus scored 13.35%. Neither model retrieves its sentence above chance and noise. These models report no mixed-task score, so they do not repeat the GLIM pattern. [Source](research_results/public_outputs_check_v8_20261112/REPORT.md).

## 5. Random splits and list position

ZuCo readers saw sentences in one fixed order, and EEG drifts over a session. On a random sentence split, the neighbours of a held-out sentence are usually in training.

**Neighbouring-text baseline.** With no EEG, predicting a held-out sentence as the text of training sentences at its true list position scores 51.09% at 10-way on average over 20 random 10% splits (range 38.36 to 65.62), and 43.73% at 25% holdout. Our own CNN on a random split scored 21.10% on the same kind of query. [Source](research_results/position_bridge_v4_20261105/REPORT.md).

**EEG encodes list position.** A ridge model from EEG to list position, for readers seen in training, predicted held-out sentences' positions with a correlation of 0.80 (ZuCo 1.0) and 0.88 (ZuCo 2.0). Matched noise, which keeps each observation's channel amplitudes, still reached 0.54 and 0.69. Copying the text of training sentences at the predicted position scored 15.41% at 10-way, against 9.70% with shuffled positions (gap 5.72, reader interval 3.29 to 8.23). Position is coarse here (off by 21 to 37 sentences), so this route reaches about half of the CNN's gain above chance. It shows the route exists, not how much the CNN used it.

**The split changes the score.** The same CNN architecture scored 23.10% at 10-way on validation under a random split and 10.02% under a block split with buffers (matched noise 9.72%). That drop also reflects fewer training sentences and different preprocessing, so it does not measure the leak alone. [Source](research_results/block_holdout_pilot_v2_20261105/REPORT.md).

## 6. Frozen tests on held-out blocks

### 6.1 Design

Each of 13 list blocks (5 in ZuCo 1.0, 8 in ZuCo 2.0) is held out once. Training leaves out that block and 10 list positions on each side. This scores 255 sentences. Each test had a written plan, a configuration file whose SHA-256 was recorded in the run script before running, synthetic tests showing that position-only and duration-only signals could not pass, a leakage audit that refused to run on any failure, and a stop rule. Controls:

| Control | What it rules out |
|---|---|
| Matched noise | Channel amplitudes and slow drift; keeps each observation's amplitude (and, for word features, its fixation pattern), removes the specific waveform |
| Shuffled pairs | Pipeline artefacts; breaks the EEG–sentence pairing |
| List position, no EEG | Place in the list, even inside a block |
| Timing and word count, no EEG | Sentence length and reading time |
| Gaze and word properties, no EEG | What eye tracking alone carries |
| Partial check | Removes position, its square, word count and reading time from real and noise predictions inside each block |

### 6.2 Results

**Table 3.** Expected 10-way within the query's own held-out block, 255 sentences, all readers averaged. Chance 10%. Word-locked rows use centred scoring, so they are not on exactly the same scale as the whole-sentence row.

| Test | Real EEG | Matched noise | Shuffled pairs | Strongest no-EEG control | Real minus noise (sentence 95%) |
|---|---:|---:|---:|---:|---|
| Whole-sentence, reader-averaged (v5) | 11.25 | 12.13 | 10.18 | Timing 12.07 | −0.88 (−3.01 to 1.19) |
| Word-locked band features (v6) | 14.55 | 16.25 | 10.37 | Gaze 26.05 | −1.70 (−6.43 to 2.81) |
| First-fixation waveform (v9) | 12.81 | 9.72 | 10.11 | Gaze 26.05 | 3.08 (−1.50 to 7.79) |

Sources: [v5](research_results/averaged_eeg_v5_20261106/REPORT.md), [v6](research_results/word_locked_v6_20261106/REPORT.md), [v9](research_results/first_fixation_v9_20261114/REPORT.md).

- **No test was clear against noise.** Real beat shuffled pairs clearly only for word-locked features (4.18 points), which shows the pipeline finds structure, but noise matched it.
- **Gaze is the strongest arm.** Gaze and word properties with no EEG beat word-locked EEG by 11.51 points (interval −16.60 to −6.34). Adding the first-fixation waveform to gaze gave 24.39%, against 24.29% for adding noise.
- **The first-fixation gap is early.** It sits 50–200 ms after fixation onset, where the early visual response appears, and fits a response to word length or appearance that gaze already captures. Removing 21 anterior channels nearest the eyes left a 4.54-point gap, not clear across readers, with noise falling to 7.35%.
- **Structure explains the word-locked gain.** Removing position, word count, reading time and fixated-word count took word-locked real EEG to 10.02%.
- **Averaging readers does not help EEG more than noise.** At one reader, real and noise were 11.00 and 11.05 (v5), 11.72 and 11.96 (v6), and 10.29 and 10.13 (v9).

### 6.3 New readers

Fitting on 15 readers and averaging the other 15 on the same held-out blocks gave 9.67% for whole-sentence features (noise 11.23%) and 13.22% for word-locked features (noise 13.31%).

### 6.4 The same sentence across releases

Forty-eight sentences were read in both releases by different readers. With nothing fitted, a sentence's averaged EEG found its own copy in the other release 28.18% of the time at 10-way, against 26.68% for matched noise (gap 1.50, interval −4.98 to 8.20). List position alone scored 54.39% and word count with reading time 52.74%, because the shared sentences sit in similar positions in both lists (correlation 0.70). With these removed, real scored 18.40% and noise 16.06%. [Source](research_results/same_sentence_v7_20261106/REPORT.md).

## 7. Discussion

**What the scores measure.** On GLIM's outputs, the mixed-task score mostly measures corpus separation. On random splits, list neighbours and EEG-predicted position can supply much of a retrieval score without sentence content. Under block holdout, every EEG representation we tried fell to the level of matched noise, and gaze with no EEG beat all of them.

**Recommendations for ZuCo retrieval evaluation.**

1. Report N-way scores within each task, and 2-way scores by same-task and different-task distractors.
2. Hold out whole list blocks with buffers, not random sentences, and report a neighbouring-text baseline for any random split.
3. Use matched noise that keeps per-observation amplitude, not only Gaussian noise or chance, as the bar.
4. Report a gaze and word-property baseline with no EEG, and the gain of EEG over it.
5. Report scores against the number of readers averaged, down to one.

**What this does not show.** It does not show that ZuCo EEG holds no language information. Our models were linear or small, on summary features, and a null under these conditions is weak evidence. Raghavan et al. (2025) find word features in fixation-locked ZuCo EEG, which is consistent with our early visual gap.

## 8. Limitations

- **Reused data.** The 255 block-holdout sentences were all used in earlier development, and the first-fixation lead was found on them. These results are exploratory. The reserved v2 test blocks have not been scored. **TODO:** decide whether to score them once, under a new frozen plan, before submission.
- **Small samples.** 255 sentences in 13 blocks, 48 shared sentences and 79–156 output sentences give wide intervals. Small effects (up to about 2–3 points over noise) are not excluded.
- **Only GLIM shows the mixed-task pattern.** SemKey's outputs are not public, so the strongest published model is untested.
- **Content or block.** The corpus gap in GLIM's outputs could reflect content or recording session. The same-sentence test covers biography sentences only and does not settle this.
- **Noise definition.** Our matched noise keeps per-observation amplitude, which makes it a strong null that sits above chance. A different null could give different gaps.
- **Inherited preprocessing.** We used the supplied preprocessed EEG and could not refit upstream filtering or referencing.

## 9. Reproducibility and ethics

Code, frozen plans, configuration hashes, leakage audits and numeric results are in the public repository (**TODO:** add the URL and a release tag after the owner's decision). No raw EEG or ZuCo text is redistributed; readers must obtain ZuCo from its official source under its terms. **TODO:** confirm the ZuCo licence terms for any derived files and for quoting sentences.

## References (to complete)

- Hollenstein, N. et al. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading. *Scientific Data*. **check**
- Hollenstein, N. et al. (2020). ZuCo 2.0: A dataset of physiological recordings during natural reading and annotation. *LREC*. **check**
- Hollenstein, N. et al. (2023). The ZuCo benchmark on cross-subject reading task classification with EEG and eye-tracking data. PMC9878684. **check** title
- Li, R. et al. (2020). The perils and pitfalls of block design for EEG classification experiments. *IEEE TPAMI*. doi:10.1109/tpami.2020.2973153
- Hollenstein, N. et al. (2021). Decoding EEG brain activity for multi-modal natural language processing. arXiv 2112.06310. (Quotes verified.)
- Wang, Z. and Ji, H. (2022). Open vocabulary electroencephalography-to-text decoding and zero-shot sentiment classification. *AAAI*. **check**
- Jo, H. et al. (2024). Are EEG-to-text models working? **check** venue.
- GLIM. **check** full citation; outputs from github.com/justin-xzliu/GLIM, commit e1f202c.
- SemKey. **check** full citation.
- Zhang et al. (2026). *ACL*. https://aclanthology.org/2026.acl-long.61/ **check** authors and title.
- McGuire, J. and Moshfeghi, Y. (2025). *SIGIR*. arXiv 2412.06695. **check**
- Marquardt et al. (2026). arXiv 2608.20186. **check**
- Raghavan et al. (2025). PMC12205611. **check**
- Public repositories: bleepbloop301/eeg2text-reproductions, RahmaDawy/eeg-language-decoding, thestonedape/task-aware-eeg2text (accessed 2026-11-06).

## Open decisions before submission

1. Venue and length (workshop short paper, or a findings or dataset-and-benchmarks track).
2. Whether to score the reserved v2 test blocks once as confirmation.
3. Whether to contact the authors of the overlapping repositories.
4. Whether to wait for SemKey's outputs.
5. Supervisor's answers to the questions sent on 6 November.
