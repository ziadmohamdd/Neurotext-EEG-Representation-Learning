# Public outputs check v8: results

Run on 2026-11-12 with the frozen plan in `configs/public_outputs_check_v8_20261112.json` (SHA-256 `7b3f2ebdf646d9b812c78ab8aecf265f9ccee88e5ebf0115b8666344f1daf0f9`). No EEG was read and nothing was trained.

**Result: neither model's generated text retrieves its sentence above chance and noise.** This covers the BART model of [Wang and Ji 2022](https://github.com/MikeWangWZHL/EEG-To-Text) and the T5 version of [Jo et al. 2024](https://github.com/NeuSpeech/EEG-To-Text), on their own published test outputs. Both earned the label "No retrieval above chance or noise in these outputs".

**This is not the GLIM pattern.** GLIM's outputs scored 21.46% when tasks were mixed and fell within a task. These two models have no mixed-task score to explain. The T5 version shows a weak hint in GLIM's direction, which did not meet the bar.

These are exploratory scores on 79 sentences, with no final test data.

## Primary scores

Expected 10-way, chance 10%, on 1,072 rows and 79 sentences per model (41 movie reviews and 38 biographies). Mixed compares each sentence with all 79, and within corpus compares it with its own corpus only.

| Output | Mixed | Within corpus | Movie reviews | Biographies |
|---|---:|---:|---:|---:|
| BART, EEG input | 10.01 | 10.10 | 10.13 | 10.06 |
| BART, noise input | 10.09 | 10.21 | 10.29 | 10.13 |
| T5, EEG input | 12.38 | 9.53 | 9.15 | 9.94 |
| T5, noise at test (`EEG_noise`) | 9.61 | 10.35 | 10.31 | 10.38 |
| Corpus label only, no EEG | 13.35 | 5.49 | 6.28 | 4.65 |

The label-only predictor is the mean text of the other test sentences in the query's corpus. It beats both models on the mixed pool by knowing only the corpus. The table shows the BART run's values, and the T5 run's differ by at most 0.07. It falls below chance within a corpus because it is built from the distractors.

## Decision

Each row needs a gap of at least 2 points and a 95% lower bound above 0, from resampling sentences.

| Comparison | BART | Clear | T5 | Clear |
|---|---:|---|---:|---|
| Mixed minus chance | 0.01 (-4.31 to 5.02) | No | 2.38 (-0.65 to 5.63) | No |
| Mixed minus noise | -0.08 (-0.35 to 0.15) | No | 2.77 (0.69 to 5.23) | Yes |
| Within corpus minus chance | 0.10 (-4.43 to 5.25) | No | -0.47 (-3.06 to 2.46) | No |
| Within corpus minus noise | -0.11 (-0.32 to 0.07) | No | -0.82 (-2.99 to 1.41) | No |
| Mixed minus within corpus | -0.09 (-4.82 to 4.11) | No | 2.85 (0.70 to 4.88) | Yes |

- **BART:** its outputs barely depend on the input. The 1,072 scored rows hold 4 distinct outputs, and one of them fills 82.00% of rows. With noise input there are 3, and one fills 90.11%.
- **T5:** the mixed score beat noise but not chance, so the label is the first one. The label needed both.

## The hint in the T5 outputs

None of this changes the label.

| 2-way | Same corpus | Different corpus |
|---|---:|---:|
| T5, EEG input | 48.76 | 59.85 |
| T5, noise at test | 49.39 | 49.52 |
| BART, EEG input | 50.03 | 50.16 |

With EEG input, the T5 outputs pick the right sentence 59.85% of the time when the distractor is from the other corpus, and 48.76% within a corpus. Its within-corpus 10-way score is 9.53%. This is the same direction as GLIM, where the only signal was which corpus a sentence came from, but it is far smaller (GLIM scored 72.83% against different-task distractors) and has no interval here. Treat it as a loose end, not a finding.

## Other secondary results

| Output | 2-way | 4-way | 10-way | 24-way |
|---|---:|---:|---:|---:|
| Chance | 50.00 | 25.00 | 10.00 | 4.17 |
| BART, EEG input | 50.10 | 25.06 | 10.01 | 4.20 |
| BART, noise input | 50.19 | 25.12 | 10.09 | 4.22 |
| T5, EEG input | 54.37 | 29.26 | 12.38 | 5.48 |
| T5, noise at test | 49.45 | 24.13 | 9.61 | 4.11 |
| T5, `noise_EEG` | 50.80 | 26.44 | 11.40 | 5.29 |
| T5, `noise_noise` | 51.25 | 26.37 | 10.81 | 4.62 |

All mixed-pool scores. The meaning of the T5 file names is not documented.

**Teacher-forced outputs retrieve the sentence equally well from noise.** These outputs are conditioned on the target text, so their scores say nothing about EEG:

| Teacher-forced, mixed 10-way | EEG input | Noise input |
|---|---:|---:|
| BART | 81.96 | 81.27 |
| T5 | 77.58 | 77.42 |

This restates in retrieval terms what [Jo et al.](https://arxiv.org/abs/2405.06459) already showed with generation metrics. The BART `generation_results-7_22` file was skipped, because its rows did not line up with the plain file.

## What this does and does not show

- **Shown:** on their own published outputs, two more ZuCo models show no sentence retrieval above chance and noise, mixed or within a corpus.
- **Shown:** a predictor that knows only the corpus scores 13.35% on the mixed pool, above both models.
- **Not shown:** that the GLIM pattern is general. These models have no mixed-task score to break down, so this neither repeats nor contradicts the GLIM finding.
- **Not shown:** anything about these models' training, or about SemKey.
- **Limits:** 79 sentences. Intervals come from sentences only, as the files carry no reader labels. Corpus labels are by elimination, and only 6 of 104 sentences could be checked against GLIM's labels. It is not known which runs produced the files. 25 protected biography sentences were removed, so the biography pool is smaller than the models' own test set.

## Integrity

- **No EEG file was opened** and no final test data was scored.
- **Audit:** no failures. File sizes and hashes, targets, counts and corpus labels matched the plan.
- **Protected data:** [2,128 prior result and manifest files were unchanged](preservation_after.json).
- **Files:** the 9 downloaded files (4.71 MB) are in `data/external/eeg_to_text_outputs_20261112/`, kept out of git. The files here hold numbers only, with no ZuCo text or model outputs.
- **Tests:** 360 passed and 2 were skipped before freezing. Synthetic random, corpus-only and sentence-level outputs earned their labels on 20 of 20 seeds each.

Files: [decision.json](decision.json), [secondary.json](secondary.json), [leakage_audit.json](leakage_audit.json), [plan.json](plan.json). [The plan](../../public_outputs_check_plan_v8_20261112.md) lists changes made before freezing.
