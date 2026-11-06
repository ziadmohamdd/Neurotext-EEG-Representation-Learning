# Systematic literature search: EEG-to-text retrieval on ZuCo

Search date 2026-11-06. Protocol written before searching. Counts are in [search_log.json](search_log.json), records in [records.csv](records.csv), and the search script in [scripts/search_literature_20261106.py](../../scripts/search_literature_20261106.py).

## The answer

| Claim | Verdict in papers | Also in public code, not peer reviewed | What is left for us |
|---|---|---|---|
| (a) Mixed-task N-way scores mostly measure telling corpora or tasks apart | **Partly published.** The parts exist. No paper states the claim | **Yes**, two repositories state it directly | The per-task and 2-way breakdown of a published model's own outputs |
| (b) Random sentence splits are weak; list position or block drift can leak | **Partly published.** EEG decoding recording blocks on ZuCo is published. The no-EEG neighbouring-text baseline was not found in a paper | **Yes**, trial-position-only baselines on ZuCo | The neighbouring-text baseline, and position correlation on held-out sentences |
| (c) Under whole-block holdouts with buffers, EEG does not beat matched noise | **Partly published.** "EEG matches noise on ZuCo" is published. The block-with-buffers test was not found in a paper | **Partly**, sentence-level only, without buffers | Buffers, reader-averaged EEG and word-locked EEG, in one frozen design |
| (d) Gaze and word properties with no EEG beat word-locked EEG | **Partly published.** Published for task classification, not for sentence retrieval | **Yes** for retrieval on ZuCo, but not with time-locked word EEG | The time-locked word-level comparison |

**No claim is "already published" as a whole, and none is fully "not found".** Not found does not mean it does not exist.

**What this means for novelty.** A paper is still possible, but the idea is no longer open ground.

- The closest paper is [Zhang et al., ACL 2026](https://aclanthology.org/2026.acl-long.61/). It already says ZuCo scores match noise and blames EEG instability. It does not test splits by list position.
- The biggest threat is not a paper. It is [bleepbloop301/eeg2text-reproductions](https://github.com/bleepbloop301/eeg2text-reproductions), a repository last pushed on 2026-10-29. It tests GLIM-style retrieval and corpus scores against session, position and length controls on ZuCo, and reaches the same conclusion as (a) and most of (b). I found no paper from it. It could become one.
- Other repositories cover nearby ground: [RahmaDawy/eeg-language-decoding](https://github.com/RahmaDawy/eeg-language-decoding) for (c) and (d), and [thestonedape/task-aware-eeg2text](https://github.com/thestonedape/task-aware-eeg2text) for (a).
- What looks new is the combination, not any single idea: published model outputs broken down by task, a position baseline, frozen block holdouts with buffers, matched noise across sentence, averaged and word-locked EEG, and a gaze baseline. A paper would have to cite these repositories as prior work.

## Second check of the key sources

A second pass reread the works that decide novelty, directly from their own pages, on 2026-11-06.

| Source | Result of the check |
|---|---|
| [Hollenstein et al. 2021](https://ar5iv.labs.arxiv.org/html/2112.06310) | Confirmed: 14 ZuCo 2.0 blocks, errors mostly in neighbouring blocks, impedance checks as the suggested cause, and the gaze and EEG medians. The text gives no exact block accuracy, and the abstract does not mention blocks |
| [McGuire and Moshfeghi](https://arxiv.org/html/2412.06695v3) | Confirmed: SIGIR 2025, P@5 12.39 vs 4.13 noise, leave-one-subject-out 14.1. Noise is listed as a test-time query condition, and the paper does not say it was used in training. So it is a weaker null than ours |
| [bleepbloop301, A3 note](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/references/a3_glim_semantics.md) | Confirmed: 67.1 falling to 47.8, 8.4 and 8.5. These come from its own linear reimplementation. **It also audits GLIM's released outputs:** a keyword rule recovers the corpus from the generated text 97% of the time. That is closer to our claim (a) than the table above says, though it gives no within-task N-way breakdown |
| [thestonedape](https://github.com/thestonedape/task-aware-eeg2text) | Confirmed: about 95% of the margin, with no baseline values. The README does not name the audited system. It mentions a working manuscript in the repository |
| Repository dates, from the GitHub API | bleepbloop301 created 2026-10-08, last pushed 2026-10-29. RahmaDawy last pushed 2026-10-27. thestonedape last pushed 2026-07-29 |

**One gain for us.** Hollenstein et al. 2021 state that block and sentence order was identical for all subjects. This supports the list-order assumption in our tests, though it does not by itself confirm that our sentence IDs follow that order.

The other works were not rechecked this way, so check them before citing.

## Protocol

**Research questions.** For each claim: has anyone shown it on ZuCo or closely similar reading-EEG data? Plus: does anyone contradict it, or report above-noise decoding under a strict split?

**Databases.** OpenAlex (primary), arXiv API, ACL Anthology (web search limited to aclanthology.org), Semantic Scholar graph API, Crossref. No email or personal data was sent. The exact query strings are in the script and the log.

**Citation tracing.** Forward and backward through OpenAlex and Semantic Scholar, from the nine seeds in the brief. I added the two ZuCo dataset papers as seeds, as anything using ZuCo should cite them.

**Inclusion.** Decodes, classifies or retrieves language content from EEG or MEG during reading. Or critiques evaluation of such decoding. Or studies leakage, block, autocorrelation or drift confounds in EEG decoding (as background). Or compares eye tracking with EEG in reading. Any language, any year, papers and preprints alike.

**Exclusion.** No brain signal, or no link to reading, language or decoding evaluation. Clinical, BCI speller, motor imagery and imagined-speech work with no bearing on the claims.

**Deviations from the protocol.** Both were added during the search and are logged.

1. General web searches, as they found work the scholarly databases missed. [Reading Task Classification](https://arxiv.org/abs/2112.06310)'s block analysis, for one, is not visible from its title or abstract.
2. GitHub repository search, after a web search turned up a code-only negative result.

## Search and flow

Queries with recorded counts: OpenAlex 24, arXiv 13, Semantic Scholar 8 (3 of them rate-limited with HTTP 429 on both tries), Crossref 4, ACL Anthology 4 (no totals available), web 5, GitHub 3.

| Stage | Records |
|---|---|
| Found, across all queries and citation traces | 4,632 |
| Unique after removing duplicates by DOI, arXiv id and title | 2,550 |
| Removed by keyword rules 1 and 2: no brain or eye signal term, or no language or evaluation term, in title or abstract | 1,084 |
| Tagged by keyword rule 3: title clearly off topic | 296 |
| Titles read and judged | 1,170 |
| Abstracts read | about 70 |
| Full text fetched and read | 18 papers, plus READMEs and result notes of 5 repositories |
| Included in the evidence | 63 records (some are versions of one work): 24 full text, 34 abstract only, 5 code |

I spot-checked 50 random titles removed by rules 1 and 2. None was close to our claims.

### Seeds

All nine resolved, and so did both extra seeds. Counts are records returned.

| Seed | OpenAlex id | OpenAlex cited-by | OpenAlex refs | S2 cited-by | S2 refs |
|---|---|---|---|---|---|
| [ZuCo benchmark, Hollenstein 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/) | W4315651769 | 31 | 83 | 17 | 81 |
| [Zhang, ACL 2026](https://aclanthology.org/2026.acl-long.61/) | W7166799362 | 0 | 6 | 0 | 55 |
| [Rethinking splits, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.289/) | W4416036958 | 0 | 2 | 6 | 29 |
| [Jo et al.](https://arxiv.org/abs/2405.06459) (merged with Scientific Reports version) | W4416879689 | 23 | 18 | 52 | 14 |
| [Beware](https://arxiv.org/abs/2405.17024) (merged with a published version) | W4415224111 | 9 | 65 | 16 | 57 |
| [GLIM](https://arxiv.org/abs/2505.17099) | W6967149031 | 0 | 0 | 8 | 61 |
| [SemKey](https://arxiv.org/abs/2603.03312) | W7133744529 | 1 | 0 | 0 | 55 |
| [Brain-CLIPLM](https://arxiv.org/abs/2604.16370) | W7171612468 | 1 | 31 | 2 | 42 |
| [Decoding silent reading](https://arxiv.org/abs/2608.20186) | W7203955387 | 0 | 0 | 1 | 25 |
| [ZuCo 1.0](https://www.nature.com/articles/sdata2018291) (extra) | W2905110202 | 200 of 202 | 22 | 259 | 24 |
| [ZuCo 2.0](https://arxiv.org/abs/1912.00903) (extra) | W3029724270 | 48 | 26 | 139 | 32 |

OpenAlex has thin data for 2026 preprints, so Semantic Scholar did most of that tracing.

## Closest works

"Noise" below means noise in place of EEG. Whether it was used at training or only at test matters, as our matched-noise arms train on it.

| Work | Data | Split | Controls | Finding | Overlap with us |
|---|---|---|---|---|---|
| [Zhang et al., ACL 2026](https://aclanthology.org/2026.acl-long.61/) | ZuCo 1.0/2.0, and new COFETT inner-speech corpus | Different sessions for COFETT. ZuCo split not stated in the paper | Noise at train and test; with and without teacher forcing | ZuCo embedding Pearson: BrainBART 5.98 EEG vs 5.96 noise, EEGPT 7.04 vs 6.51. COFETT 14.92 and 17.13 | (c) headline. No list-position or block analysis on ZuCo |
| [Jo et al.](https://arxiv.org/abs/2405.06459) | ZuCo | Unique sentences, 80/10/10 | Noise crossed at train and test | BART no teacher forcing BLEU-1: 13.69 EEG, 14.22 noise | (c), for generation. Teacher forcing is their main point |
| [Yin et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.289/) | Narratives fMRI, ZuCo 1.0 | Reader and text disjoint | Shuffled input, extra test set | Text overlap makes shuffled input score the same. Their strict split: EEG2Text BLEU-1 24.49 vs 18.72 shuffled | (b) in spirit. Covers reader and text overlap, not order |
| [Hollenstein et al. 2021](https://arxiv.org/abs/2112.06310) | ZuCo 1.0, 2.0 | Within-subject and leave-one-subject-out | Text baselines, block ablation | ZuCo 2.0 EEG electrode features classify 14 recording blocks well, and errors fall in neighbouring blocks | **(a), (b), (d).** Task classification, not retrieval |
| [Hollenstein et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/) | ZuCo 2.0 train, new held-out readers | Held-out subjects, shuffled test | Random, BERT, text difficulty | Accuracy: eye tracking 0.69, EEG 0.55 to 0.58, BERT 0.65 | **(d)** for task classification, (b) block risk known |
| [GLIM](https://arxiv.org/abs/2505.17099) | ZuCo | Unique sentences, 8:1:1 | Gaussian noise at test | Retrieval among 24 within task, subject, dataset, corpus: 8.15 vs 3.67 noise. Corpus 93.48 vs 62.73 | (a): reports corpus score as success |
| [SemKey](https://arxiv.org/abs/2603.03312) | ZuCo | Unique text, stratified | Gaussian noise at test | 10-way 29.56, GLIM 21.91. Noise falls to chance | (a): trials are not limited to one task |
| [Brain-CLIPLM](https://arxiv.org/abs/2604.16370) | ZuCo | Unique sentences 80/10/10, or leave one subject out | Random and oracle anchors | Top-5 among 300 to 407 sentences: 67.6, random anchors 12.3 | Contradicts (c) on its face. See below |
| [McGuire and Moshfeghi, SIGIR 2025](https://arxiv.org/abs/2412.06695) | ZuCo | 5-fold 80/10/10; leave one subject out | Matched-statistics noise | P@5 12.39 vs 4.13 noise; leave-one-out 14.1 | Contradicts (c) on its face. See below |
| [ETER](https://arxiv.org/abs/2408.04679) | ZuCo | Not clearly stated | Random model | Word top-1 8.66 vs 1.08 random | Unseen EEG segments, not shown to be unseen words. No noise training |
| [RAG pipeline](https://arxiv.org/abs/2605.17503) | ZuCo 1.0, 9 readers | Within reader, 650/50, same 50 test sentences | EEG shuffled in time | Cosine 0.181 vs 0.139 | Weak control, no list-position check |
| [BrainMosaic](https://arxiv.org/abs/2601.20447) | ZuCo and others | Within reader, random 8:2 | Label and prediction random baselines | Reports wins over baselines | Random split, no noise training |
| [Marquardt et al.](https://arxiv.org/abs/2608.20186) | Own, 1 reader, 49 h, 19 channels, single-word display | Whole runs held out | Permutation, run-mate swap, position probes | Within-run top-10 gain 6.7 points, 7.5 with context-free targets | **Strict positive result.** Not ZuCo |
| [d'Ascoli et al., Nature Communications 2025](https://arxiv.org/abs/2412.17829) | 723 readers and listeners, with single-word display in reading | Sentence hashed to one split | Linear baselines, chance | Reading better than listening. Top-10 of 250 words, above chance for every subject | Strict by sentence, not by order |
| [Raghavan et al., iScience 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205611/) | ZuCo 1.0, 2.0 | Encoding model | Cluster permutation | Fixation-locked EEG tracks word features, including surprisal | Real word-locked signal on ZuCo. Encoding, not retrieval |
| [Beware](https://arxiv.org/abs/2405.17024) | Watermelon "phantom" EEG, and human | Same or different domain | Non-neural source | Watermelon EEG reaches 88.78 to 99.99 on domain labels, chance 2.5 to 12.5 | (b) in general, not reading |
| [Zhang X. et al. 2026](https://arxiv.org/abs/2605.24524) | MEG, EEG, listening | Stimulus identity | Noise, fixed windows | Noise gets 66.3 by a duration shortcut. Brennan EEG near chance | (b), (d) in spirit. Not reading |
| [BrainECHO](https://aclanthology.org/2025.findings-acl.142/) | Listening EEG, MEG | Session, sentence, subject | Noise | Scores far above noise, high across splits | Not reading |
| [bleepbloop301 repo](https://github.com/bleepbloop301/eeg2text-reproductions), code | ZuCo, COFETT | Random and blocked | Noise, surrogates, length, gaze, position | See below | **(a), (b), (c), (d).** Not reviewed |
| [RahmaDawy repo](https://github.com/RahmaDawy/eeg-language-decoding), code | ZuCo | Unseen readers and sentences | Random input, length, behaviour | EEG at chance. Behaviour beats EEG | **(c), (d).** Not reviewed |
| [thestonedape repo](https://github.com/thestonedape/task-aware-eeg2text), code | ZuCo, frozen GLIM | Grouped by sentence | Matched wrong-real EEG | See below | **(a).** README only |

I read the data, split, control and result sections of each paper from the extracted full text, not every line.

## Evidence by claim

### (a) Mixed-task scores measure corpus or task

**Partly published.**

- [GLIM](https://arxiv.org/abs/2505.17099) reports corpus classification of 93.48 and retrieval of 8.15 within groups of one task and corpus. It reads the corpus score as semantic decoding, not as a confound.
- [SemKey](https://arxiv.org/abs/2603.03312) draws its N-way trials from random negatives and reports GLIM at 21.91. [Our earlier check of its code](../published_splits_check_20261105.md) found no limit to one task.
- [Hollenstein et al. 2021](https://arxiv.org/abs/2112.06310) get most ZuCo 1.0 readers to 99% task accuracy within subject from electrode features, but 48% across subjects. They test session as an explanation, and find adding a third task barely changes EEG accuracy.
- **Not found in a paper:** anyone showing a mixed N-way score falling when distractors come from the same task.

Code, not reviewed:

- [bleepbloop301, A3 note](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/references/a3_glim_semantics.md): task identity alone scores 100% on corpus. A linear model's EEG corpus score of 67.1 falls to 47.8 once session offsets are removed. It concludes that GLIM's 93% measures session.
- [thestonedape](https://github.com/thestonedape/task-aware-eeg2text): the README says matching distractors by dataset and task removes about 95% of a released system's above-chance Top-1 margin. The manuscript is not public, so I could not check this.

### (b) Random splits and list position

**Partly published.**

- [Hollenstein et al. 2021](https://arxiv.org/abs/2112.06310) is the owner of the block finding. In ZuCo 2.0, within subject, electrode features classify the 14 blocks well, and mistakes land mostly on neighbouring blocks. They suggest impedance checks every 3 or 4 blocks as the cause. They also show that task classification benefits from block information. They note that recording order was the same for all readers. **This is prior art for "EEG predicts where in the list you are", and our earlier checks missed it.**
- [Hollenstein et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/) shuffled test sentences to stop models trying to "identify an experimental block". That is a task-classification benchmark. The risk is known to ZuCo's authors.
- General EEG: [Li et al.](https://doi.org/10.1109/tpami.2020.2973153) and [Beware](https://arxiv.org/abs/2405.17024) on block designs, disputed by [Palazzo et al.](https://arxiv.org/abs/2012.03849). I read the abstract only for those two critics.
- Same logic, other modalities: [Removing Timing Shortcuts](https://arxiv.org/abs/2609.40359) reports 22.0 on synthetic signals vs 22.3 on real brain data for a leading word decoder (abstract only). [Stimulus overlap confounds](https://doi.org/10.64898/2026.10.07.757228) reports overlap explains 77% of seen-stimulus performance on one emotion set (abstract only).
- [Marquardt et al.](https://arxiv.org/abs/2608.20186) measure a positional prior with probes, on their own data.
- **Not found in a paper:** the no-EEG neighbouring-text baseline, or the 0.80 to 0.88 position correlation on held-out sentences.

Code, not reviewed: [bleepbloop301, A1 results](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/results/a1_contrastive/results.md). On 10-way retrieval within reader and task, under a random sentence split, trial position alone gets 22.7 (EEG 23.4). In length-matched pools position gets 19.8 and beats EEG at 18.1.

**A point not stated in any paper I read.** The [EEG-To-Text code](https://github.com/MikeWangWZHL/EEG-To-Text/blob/main/data.py) splits by sequence, with the first 80% of unique sentences for training and the last 10% for testing, and no buffer. [An audit](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/references/a1_reproducibility_audit.md) says Jo et al. used the same rule. So their noise null may come from an end-of-list holdout, if the data order follows presentation order, which I did not check. [GLIM](https://arxiv.org/abs/2505.17099) moved to a random split and reports a gap above noise. This is a question worth testing. It is not a result.

### (c) Block holdouts against matched noise

**Partly published.**

- [Jo et al.](https://arxiv.org/abs/2405.06459) and [Zhang et al.](https://aclanthology.org/2026.acl-long.61/) publish "EEG matches noise" on ZuCo, trained on noise, mostly for generation. Zhang adds an embedding Pearson metric.
- **Not found in a paper:** whole-block holdouts with buffers on ZuCo, reader-averaged EEG against matched noise, or word-locked EEG against matched noise under such a split.

Code, not reviewed:

- [bleepbloop301, A2 note](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/references/a2_cofett_vs_zuco.md): ridge from sentence-level band power to LaBSE on ZuCo 1.0 movie reviews. Rank 0.500 on random folds and 0.501 on blocked folds. Chance is 0.500. No buffers are mentioned. Its dashboard has a block split for contrastive retrieval, but I found no saved result and did not run it.
- [RahmaDawy](https://github.com/RahmaDawy/eeg-language-decoding): 45 candidates, chance top-1 2.2%, unseen readers and sentences, ridge to MiniLM. Random input 2.2%, band power 3.1%. Its limits section calls time-locked word-level decoding the main untested direction. That is the ground of our word-locked arm. It compares with random input, and uses no block holdout.

### (d) Gaze beats EEG

**Partly published.**

- [Hollenstein et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/): held-out readers, task classification. Eye tracking 0.69, EEG 0.55, 0.58 with PCA, and eye tracking plus EEG means 0.68.
- [Hollenstein et al. 2021](https://arxiv.org/abs/2112.06310): across subjects, gaze median 70% on ZuCo 1.0 and 58% on ZuCo 2.0, EEG electrode features 48% and 52%.
- [Brain-CLIPLM](https://arxiv.org/abs/2604.16370) names fixation durations as part of the measurement protocol and flags this as a limit. It does not run a gaze-only arm.
- **Not found in a paper:** gaze against EEG for sentence retrieval on ZuCo.

Code, not reviewed:

- [RahmaDawy](https://github.com/RahmaDawy/eeg-language-decoding): word-level GRU, unseen sentences. MRR above chance: behaviour only 0.0803, EEG 0.0035, random 0.0015. It uses band features at first fixation, not time-locked EEG.
- [bleepbloop301, A1](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/results/a1_contrastive/results.md): random pool top-1: eye movements 23.6, length 23.5, EEG 23.4. In length-matched pools, though, EEG 18.1 beats eye movements 14.1. That goes against (d) in that setting.

## Work that contradicts us, or reports real decoding

Reported plainly. None uses our split.

| Work | Result | Why it may or may not touch us |
|---|---|---|
| [Marquardt et al.](https://arxiv.org/abs/2608.20186) | Whole runs held out, permutation null, position probes. Within-run gain 6.7 points. **Strongest strict positive result found** | One reader, 49 h, no eye movements. Upper bounds: no held-out test set, checkpoints chosen on the same metric. Suggests limits come from data volume, which our null cannot rule out. Follow-up: [Deep LLM layers](https://arxiv.org/abs/2610.23638) (abstract only) |
| [d'Ascoli et al.](https://arxiv.org/abs/2412.17829) | Reading EEG decoded well above chance across many readers, sentence-disjoint. More averaging at test raises scores, which sits against our averaged-EEG null | Single-word display, so no gaze route. I saw no order control in the sections I read. [Removing Timing Shortcuts](https://arxiv.org/abs/2609.40359) challenges part of this pipeline for speech |
| [McGuire and Moshfeghi](https://arxiv.org/abs/2412.06695) | **On ZuCo:** P@5 12.39 vs matched noise 4.13. Leave-one-subject-out 14.1. Peer reviewed | Passage retrieval on synthetic queries. The query spans come from the passage itself. I could not tell if folds keep list neighbours apart |
| [Brain-CLIPLM](https://arxiv.org/abs/2604.16370) | **On ZuCo:** sentence top-5 of 67.6, vs 12.3 for random anchors | Random sentence split. Fixed keyword vocabulary and retrieval from a task-specific corpus can favour the pool. Its random-anchor control tests the language model, not noise-trained EEG |
| [GLIM](https://arxiv.org/abs/2505.17099) | **On ZuCo:** retrieval 8.15 vs 3.67 noise, within group | Noise only at test. Random split. A linear copy gets 8.4 and [length alone gets 8.5](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/references/a3_glim_semantics.md) (not reviewed) |
| [Yin et al.](https://aclanthology.org/2025.emnlp-main.289/) | **On ZuCo:** under their strict split, ordered input beats shuffled, BLEU-1 24.49 vs 18.72 | Shuffled input, generation metric, ZuCo 1.0 only |
| [bleepbloop301, A1](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/results/a1_contrastive/results.md) | **On ZuCo:** contrastive 10-way, EEG beats noise-trained noise by 2.4 to 4.8 points. All 12 readers beat a same-length surrogate | Random sentence split. Position alone beats it in length-matched pools. Not reviewed |
| [thestonedape](https://github.com/thestonedape/task-aware-eeg2text) | **On ZuCo:** adapter over GLIM features: MRR about 0.32, chance 0.157, matched wrong-real EEG at chance | README only. GLIM features come from a random split. Not reviewed |
| [Zhang et al.](https://aclanthology.org/2026.acl-long.61/) | COFETT beats noise across sessions | Sentences repeat. In their Table 1 I can see each COFETT noise value equals the ZuCo value, which [bleepbloop301](https://github.com/bleepbloop301/eeg2text-reproductions/blob/main/references/a2_cofett_vs_zuco.md) also flags. That repository also finds trial position scores higher (not reviewed) |
| [Raghavan et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205611/) | **On ZuCo:** fixation-locked EEG tracks word features | Encoding at word level. It does not show sentence identity can be retrieved, but it means "no word-locked signal" would be the wrong way to state our result |

Our findings should be stated narrowly: no above-noise retrieval, with these models and this data volume, under this split. Not "EEG has no language signal".

## What a paper would have to cite

- **Data:** [ZuCo 1.0](https://www.nature.com/articles/sdata2018291), [ZuCo 2.0](https://arxiv.org/abs/1912.00903).
- **Noise and evaluation:** [Jo et al.](https://arxiv.org/abs/2405.06459), [Zhang et al.](https://aclanthology.org/2026.acl-long.61/), [Yin et al.](https://aclanthology.org/2025.emnlp-main.289/).
- **Blocks and task on ZuCo:** [Hollenstein 2021](https://arxiv.org/abs/2112.06310), [Hollenstein 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/).
- **Audited models:** [GLIM](https://arxiv.org/abs/2505.17099), [SemKey](https://arxiv.org/abs/2603.03312), [Brain-CLIPLM](https://arxiv.org/abs/2604.16370), [Wang and Ji](https://arxiv.org/abs/2112.02690), [McGuire and Moshfeghi](https://arxiv.org/abs/2412.06695).
- **Autocorrelation in general:** [Beware](https://arxiv.org/abs/2405.17024), [Li et al.](https://doi.org/10.1109/tpami.2020.2973153), [Palazzo et al.](https://arxiv.org/abs/2012.03849).
- **Shortcuts elsewhere:** [Removing Timing Shortcuts](https://arxiv.org/abs/2609.40359), [Zhang X. et al.](https://arxiv.org/abs/2605.24524).
- **Real decoding and counter-evidence:** [Marquardt et al.](https://arxiv.org/abs/2608.20186), [d'Ascoli et al.](https://arxiv.org/abs/2412.17829), [Raghavan et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12205611/).
- **Public code with overlapping results:** [bleepbloop301](https://github.com/bleepbloop301/eeg2text-reproductions), [RahmaDawy](https://github.com/RahmaDawy/eeg-language-decoding), [thestonedape](https://github.com/thestonedape/task-aware-eeg2text).

## Limits of this search

- **Not found is not absent.** New preprints appear weekly in this area, and several works above are months old.
- **Semantic Scholar was rate-limiting.** Queries S205, S207 and S208 returned HTTP 429 on both tries. Its citation lookups did work.
- **OpenAlex keyless budget.** It gives a small shared daily budget. I used about a third. A later page of ZuCo 1.0 cited-by returned HTTP 401, which lost 2 of 202 records. Search results are capped at the first 100 per query.
- **Google Scholar was not searched.** It has no API.
- **Screening used keyword rules first.** Rules 1 to 3 removed 1,380 records unread by me, except for a 50-title spot check. Abstracts were mostly read for shortlisted titles, not all 1,170.
- **The block finding shows why this matters.** The most important paper for (b) is invisible from its title and abstract. Others may be hidden the same way.
- **Depth of reading.** I read sections, not every line, and some numbers come from tables with broken text extraction. Check numbers against each paper before citing. The abstract-only works are marked in the text and in [records.csv](records.csv).
- **Repositories are unreviewed.** I read their notes and did not rerun anything. I could not tell whether their authors plan a paper, or whether any are related to this project's group. Worth asking before submitting.
- **Comparisons are loose.** None of these works use our split, targets and metric, so their numbers cannot be directly compared with ours.
- **Page text addressed to agents.** An OpenAlex API notice invited agents to post the user's email to obtain a key, and a GitHub issue contained an unsolicited contact line. I followed neither. No email was sent anywhere.
