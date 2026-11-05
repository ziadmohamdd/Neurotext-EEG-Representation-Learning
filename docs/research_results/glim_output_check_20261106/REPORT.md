# GLIM output check: results

Run on 2026-11-06 on the three public CSV files in [GLIM's repository](https://github.com/justin-xzliu/GLIM) at commit `e1f202c`. No EEG was read and nothing was trained. All of this is exploratory, with analyses chosen as results came in.

**Main finding: on GLIM's published outputs, almost all of the N-way score comes from telling movie reviews apart from biographies.** Mixed across tasks, the outputs score 21.46% at 10-way, close to what the SemKey paper reports for GLIM (21.91%). Within a single task they score about 9% to 14%.

**The neighbouring-sentence route did not show up in GLIM's outputs.** Neighbouring text with no EEG far outscores GLIM on the same queries, but GLIM's generated text does not move towards a sentence's list neighbours. So this check gives no support for saying GLIM's score came from that route.

## Reproducing the published score

Each score below uses SemKey-style trials: sample N rows and ask whether each output is closest to its own sentence. GLIM's CSV files hold 2,208 of its test rows. After protected sentences are removed, 1,889 rows and 156 sentences remain. Outputs and targets were embedded with all-MiniLM-L6-v2.

| Predictor | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| GLIM, as reported in the SemKey paper | 66.83 | 42.45 | 21.91 | 10.74 |
| GLIM outputs, this check | 67.26 | 42.71 | 21.46 | 10.47 |
| GLIM with noise input, this check | 51.55 | 26.50 | 11.26 | 5.06 |

The close match suggests this is the same kind of score, but it does not prove the same files or embedding model produced the published numbers.

## Where the score comes from

**2-way by distractor type.** Each sentence is compared with every other sentence, with 95% intervals from resampling sentences.

| Predictor | All pairs | Same task | Different task |
|---|---:|---:|---:|
| GLIM outputs | 68.29 | 51.75 (50.27 to 53.30) | 72.83 (68.68 to 77.03) |
| Prompt-free test outputs | 68.48 | 52.28 (50.84 to 53.88) | 72.93 (68.93 to 76.80) |
| Noise input outputs | 51.08 | 49.84 (48.58 to 51.00) | 51.42 (49.65 to 53.20) |
| Task label only, no EEG | 68.06 | 33.76 (30.13 to 36.79) | 77.48 (73.24 to 81.45) |

The label-only predictor is the mean text of the other test sentences in the query's task. Using only that label, it matches GLIM's overall 2-way score. It falls below chance within a task, because it cannot favour one sentence over another there and is built from the distractors.

**10-way, one row per sentence in each trial:**

| Pool | Sentences | GLIM | Noise input |
|---|---:|---:|---:|
| All tasks | 156 | 21.71 | 11.10 |
| ZuCo 1.0 task 1, movie reviews | 40 | 10.53 | 9.44 |
| ZuCo 1.0 task 2, normal reading | 15 | 14.13 | 11.50 |
| ZuCo 1.0 task 3, task-specific reading | 39 | 9.73 | 12.31 |
| ZuCo 2.0 task 2, normal reading | 24 | 14.25 | 11.58 |
| ZuCo 2.0 task 3, task-specific reading | 38 | 11.30 | 10.06 |

**2-way by pair of tasks** shows which distinction carries the gain:

| Pair | GLIM | Noise input |
|---|---:|---:|
| Movie reviews against any biography task (4 pairs) | 87.36 to 93.57 | 38.50 to 53.60 |
| Biography tasks against each other (6 pairs) | 51.89 to 60.66 | 49.92 to 58.18 |

## What could explain this

- **Not the task prompt alone:** ZuCo 1.0 tasks 1 and 2 share the same task and dataset prompts in GLIM's code, yet that pair scores 89.94%.
- **Real EEG is needed:** noise input scores 48.53% on the same pair.
- **Cause unknown:** the EEG could be reflecting different content, or different recording blocks. The [ZuCo authors shuffled their benchmark's test sentences](https://pmc.ncbi.nlm.nih.gov/articles/PMC9878684/) to stop models identifying a block, so this is a known risk. These files cannot separate the two.

## The neighbouring-sentence route

These use only the normal-reading list sentences we hold: 470 rows and 36 sentences after protected sentences are removed.

| Predictor | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| GLIM outputs | 53.35 | 29.85 | 14.84 | 7.92 |
| Noise input outputs | 51.19 | 27.12 | 11.81 | 5.78 |
| Neighbouring text with true position, no EEG | 86.13 | 72.68 | 52.76 | 35.84 |

Against the pool of unique sentences, 10-way is 63.30% for neighbouring text and 16.02% for GLIM. The gap of 47.28 points has a sentence interval of 31.96 to 61.75. GLIM's validation sentences cannot be identified, so some neighbours may not have been in its training set. Randomly dropping 11% of them gives 60.60%.

**Do GLIM's outputs follow the neighbours?** Mean cosine similarity to training sentences at each list distance:

| Text | Distance 1–2 | 3–5 | 6–10 | 11 or more |
|---|---:|---:|---:|---:|
| The true sentence | 0.249 | 0.204 | 0.152 | 0.119 |
| GLIM outputs | 0.164 | 0.170 | 0.178 | 0.165 |
| Noise input outputs | 0.099 | 0.096 | 0.101 | 0.099 |

The true sentence is clearly closer to its neighbours, and GLIM's outputs are flat. A position readout from the outputs was also inconsistent (correlation -0.08 in ZuCo 1.0 and 0.48 in ZuCo 2.0), and it is a weak measure, since the true text itself gave only 0.27 in ZuCo 2.0.

## Limits

- **Generated text only:** GLIM's paper also reports retrieval from EEG embeddings within 24-item groups of the same task, subject and dataset, at 8.15% against 4.17% chance. That head is not in these files, and it may hold more within-task signal than the generated text does. GLIM's own retrieval metric is therefore not affected by task mixing.
- **SemKey is not tested:** its outputs are not public. Its code samples N-way trials across the whole set it is given, so the same effect is possible, but its score may also hold more within-task signal.
- **Unknown checkpoint:** it is not known which run produced these files.
- **Prompt-free label:** I took this file's name at face value. The paper describes disabling dataset and subject prompts at test time, so the task prompt may still be active there.
- **Few sentences:** within-task pools hold 15 to 40 sentences, and the neighbour comparison holds 36. Same-task intervals are from sentences only and ignore readers.

## Integrity

- Protected sentences (v1 test, v2 test blocks and buffers) were removed by text before scoring, and from the neighbour bank.
- The three CSV files are stored in `data/external/glim_results_e1f202c/`, kept out of git, with SHA-256 starting `ec7d4fe8`, `7824a5f4` and `2e050765`. They were only parsed as data, and their sizes match the repository listing.
- Scores: [pools](pool_scores.json), [pairs](pair_scores.json), [task pairs](task_pair_scores.json) and [same-query comparison](output_scores.json). These hold numbers only, with no ZuCo text or GLIM outputs.
- Scripts: `scripts/analyze_glim_outputs.py`, `scripts/analyze_glim_pool.py`, `scripts/analyze_glim_pairs.py` and `scripts/analyze_glim_task_pairs.py`.
