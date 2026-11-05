# How published ZuCo work splits its data

Checked on 2026-11-05, against public code where it exists. Code lines for GLIM and SemKey were read directly. Paper details mostly come from automated page summaries, so read each paper before citing it.

**Finding: GLIM and SemKey hold out sentences at random, so a held-out sentence can keep its list neighbours in training.** Neither paper or repository reports a check for this. That shows the exposure exists. It does not show that these models use it or that their scores are inflated.

## What each work does

| Work | How it splits | Reading order | No-signal check |
|---|---|---|---|
| [GLIM](https://arxiv.org/abs/2505.17099), [code](https://github.com/justin-xzliu/GLIM) at `e1f202c` | Unique text. Duplicates go to training, the rest are shuffled with seed 42 within dataset and task, about 8:1:1 (17,908 / 2,200 / 2,227) | Considered in `data/__STEP4_eeg_text_match_split.ipynb`, which chooses random sampling over ordinal chunking | Gaussian noise at test time |
| [SemKey](https://arxiv.org/abs/2603.03312), [code](https://github.com/xmed-lab/SemKey) at `a3104c0` | The paper reports the same counts as GLIM, by unique text. The code differs (see below) | Not mentioned in what I read | Gaussian noise at test time |
| [Brain-CLIPLM](https://arxiv.org/html/2604.16370) | Unique sentences, 80/10/10 | Blocks not mentioned; "order" refers to word order inside a sentence | Random-anchor control |
| [ETER](https://arxiv.org/html/2408.04679v1) | Not described | Not mentioned | Random baseline |

## Details that matter

- **GLIM saw the block design and chose random sampling.** Its notebook gives two reasons: more relation types, and to "exclude potential bias caused by the 'temporal adaptability of stimuli'". Our results suggest random sampling also has a cost, as it places neighbours on both sides of the split.
- **SemKey's paper and code disagree.** The paper reports stratified sampling at about 8:1:1. `preprocess/preprocess_merge.py` uses `train_test_split` on unique text with 20% test and 10% validation, seed 42 and no stratification. I can't tell which produced the published scores.
- **SemKey's trial count also differs.** The paper says 1,000 trials and `evaluation/evaluate.py` uses 10,000. Its trials sample rows from the whole set passed in, with no limit to one release or task.
- **GLIM's retrieval is narrower.** It scores 24-item groups within task, dataset, subject and corpus. So its scores and SemKey's N-way scores are not the same measure.
- **Test-time noise cannot catch this route.** This is my reasoning, not a result. Replacing EEG at test time removes drift and content signal together, so passing that check does not separate them. Our noise arms trained on noise that keeps channel amplitudes, which is a different test.

## Neighbouring text under a similar split and metric

To see the likely size of the exposure, I scored neighbouring text with no EEG. Each row of the table is a mean over 20 random splits, or the score as published.

The local run held out 10% of sentences by unique text and used SemKey-style trials across releases, 1,000 per setting. It used normal-reading sentences only, with v1 test and v2 test-zone sentences removed.

| Predictor | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| Random training text, no EEG, local | 51.34 | 26.04 | 10.31 | 4.51 |
| **Neighbouring text with true position, no EEG, local** | **82.14** | **67.18** | **47.98** | **32.40** |
| GLIM, as reported in the SemKey paper | 66.83 | 42.45 | 21.91 | 10.74 |
| SemKey, as published | 72.29 | 50.48 | 29.56 | 16.24 |

**These rows cannot be compared directly.** The local run is not their split. It leaves out task-specific reading, which they include, and it uses our text embeddings. It also uses a sentence's true position, which no EEG model is given, and [v4](research_results/position_bridge_v4_20261105/REPORT.md) found EEG reveals position only coarsely. The table shows that this kind of split leaves a large route open, not that any published score came from it. Scores are in [the JSON file](research_results/published_style_baseline_20261105/scores.json).

## Is the idea already published?

I found no paper that measures a list-position or neighbouring-text route in ZuCo sentence retrieval. The search covered several web searches and one [OpenAlex](https://openalex.org) query, and Semantic Scholar's citation lookups were rate-limited. **It is not a systematic review, and citations have not been traced.**

The closest work, which any paper would have to cite and separate itself from:

| Work | What it covers | What is left |
|---|---|---|
| [ZuCo benchmark, Hollenstein et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/) | The ZuCo authors shuffled their held-out test sentences so that models could not "identify an experimental block" | Reading-task classification, not sentence retrieval. **This means the block risk is known to ZuCo's authors** |
| [Zhang et al., ACL 2026](https://aclanthology.org/2026.acl-long.61/) | Says ZuCo ignores EEG instability, and that noise scores match EEG scores. Builds a new corpus split by session | No analysis of list position, and no measurement of drift on ZuCo. **The closest recent critique** |
| [Rethinking Cross-Subject Data Splitting, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.289.pdf) | Reader and text leakage in splits | Presentation order is not discussed in the version I read |
| [Noise baselines, Jo et al.](https://arxiv.org/html/2405.06459v4) | Trains on noise | No split or position analysis |
| [Beware of Overestimated Decoding Performance](https://arxiv.org/pdf/2405.17024) | Temporal autocorrelation in image EEG | Not reading |
| [Decoding silent reading](https://arxiv.org/html/2608.20186v1) | Own data, with drift and position controls | Not ZuCo |

## What this means for a paper

**The route is still open, but it is narrower than "nobody knows about blocks".** The contribution would be measuring this on ZuCo sentence retrieval:

1. A no-EEG neighbouring-text baseline showing how easy random text splits are.
2. A demonstration that EEG predicts list position for held-out sentences.
3. A test of how much of a published model's score survives a split that keeps neighbours apart.

We have evidence for 1 and 2 on our own split and models. Point 3 is what would make the paper matter, and it has not been done for any published model.

## Next steps

1. **Exact test sentences:** GLIM publishes generated test outputs in three CSV files under `results/`. Those would show its real test sentences, so neighbouring text could be scored on them, for the normal-reading sentences we hold. This needs a download of about 13 MB.
2. **Real retraining:** running GLIM or SemKey on a split that keeps neighbours apart is the real test of point 3. It needs the ZuCo task-specific reading data, which isn't local.
3. **Literature:** trace citations of the works above, read them in full, and review the route with the project supervisor.
