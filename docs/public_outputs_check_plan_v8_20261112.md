# Public outputs check: plan v8

**Status: draft, not approved.** Nothing has been downloaded or scored. [The config](../configs/public_outputs_check_v8_20261112.json) is not frozen, and no run script exists yet.

## Question

The [GLIM check](research_results/glim_output_check_20261106/REPORT.md) found that GLIM's mixed-task score comes mostly from telling movie reviews apart from biographies. Does the same hold for two more published models?

- **BART model:** Wang and Ji 2022, from [the original repository](https://github.com/MikeWangWZHL/EEG-To-Text/tree/main/results).
- **T5 version:** Jo et al. 2024, from [the NeuSpeech repository](https://github.com/NeuSpeech/EEG-To-Text/tree/master/results).

**Expected result:** probably no retrieval above chance at all. Without teacher forcing the BART model repeats one sentence for many rows, and [Jo et al.](https://arxiv.org/abs/2405.06459) already report the same scores on noise. That would add two models to the audit, but as a weaker result than GLIM's.

## Download needing approval

9 text files, 4,713,314 bytes (4.71 MB) in total. They hold ZuCo sentence text, so they would go in `data/external/` and stay out of git.

| Repository | Files | Bytes |
|---|---:|---:|
| MikeWangWZHL/EEG-To-Text, `main` | 5 | 2,344,354 |
| NeuSpeech/EEG-To-Text, `master` at `a07cf91` | 4 | 2,368,960 |

The config lists each file and its size. The first repository has no licence file, and the second's licence is not verified.

## Design

1. **Input:** each model's free-generation output for EEG-trained, EEG-tested rows. No EEG is opened and nothing is trained.
2. **Corpus labels:** the files carry no task labels. A target matching a local normal-reading sentence is a biography, and every other target is a movie review. This is checked against GLIM's task labels where a sentence appears in both.
3. **Protected data:** targets matching v1 test sentences, v2 test blocks or their buffers are removed by text.
4. **Score:** expected 10-way against the pool of unique test sentences, with all-MiniLM-L6-v2, in three pools: all sentences, movie reviews only and biographies only.
5. **Controls:** each model's noise-input outputs, and a predictor that knows only the corpus label.

## Decision, per model

Each comparison needs a gap of at least 2 points and a 95% lower bound above 0, from resampling sentences.

| Result | Label |
|---|---|
| Mixed pool does not clear chance and noise | No retrieval above chance or noise in these outputs |
| Mixed pool clears both, and beats the within-corpus score | The mixed score comes mostly from telling the corpora apart |
| Mixed pool clears both, and does not beat the within-corpus score | The score holds within a corpus |

## Limits known in advance

- **Labels by elimination:** a biography sentence missing from the local lists would be counted as a movie review. The run stops if any label disagrees with GLIM's.
- **No reader labels:** intervals come from sentences only.
- **Teacher-forced outputs are secondary:** they are conditioned on the target text, so their scores say little about EEG.
- **T5 file names are undocumented:** `EEG_noise`, `noise_EEG` and `noise_noise` are reported as named.
- **Task-specific reading is left out:** the four `task1_task2_task3` files cannot be labelled with local lists.
- **Unknown checkpoints:** it is not known which runs produced these files.

## Before freezing

- Owner approves the download and this plan.
- Check the files' sizes and formats, and that they parse, without computing any score.
- Synthetic fixtures over 20 seeds each must earn the right label: random outputs, corpus-only outputs and sentence-level outputs.
- Record any change made after approval in this file.

## Stop rule

Stop after this run, whatever the outcome. No extra files, embeddings or settings under this plan. No final test data is scored.
