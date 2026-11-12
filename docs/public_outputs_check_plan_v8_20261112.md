# Public outputs check: plan v8

**Status: approved by the project owner on 2026-11-12.** [The config](../configs/public_outputs_check_v8_20261112.json) is frozen, and its SHA-256 is recorded in `scripts/run_public_outputs_check.py`. Any later change needs a new version.

## Question

The [GLIM check](research_results/glim_output_check_20261106/REPORT.md) found that GLIM's mixed-task score comes mostly from telling movie reviews apart from biographies. Does the same hold for two more published models?

- **BART model:** Wang and Ji 2022, from [the original repository](https://github.com/MikeWangWZHL/EEG-To-Text/tree/main/results).
- **T5 version:** Jo et al. 2024, from [the NeuSpeech repository](https://github.com/NeuSpeech/EEG-To-Text/tree/master/results).

**Expected result:** probably no retrieval above chance at all. Without teacher forcing the BART model repeats one sentence for many rows, and [Jo et al.](https://arxiv.org/abs/2405.06459) already report the same scores on noise. That would add two models to the audit, but as a weaker result than GLIM's.

## Downloaded files

9 text files, 4,713,314 bytes (4.71 MB) in total, downloaded on 2026-11-12 after approval. They hold ZuCo sentence text, so they sit in `data/external/` and stay out of git.

| Repository | Files | Bytes |
|---|---:|---:|
| MikeWangWZHL/EEG-To-Text, `main` at `a07cf91` | 5 | 2,344,354 |
| NeuSpeech/EEG-To-Text, `main` at `0158240` | 4 | 2,368,960 |

The config lists each file with its size and SHA-256. The first repository has no licence file, and the second's licence is not verified.

## Design

1. **Input:** each model's free-generation output for EEG-trained, EEG-tested rows. No EEG is opened and nothing is trained.
2. **Corpus labels:** the files carry no task labels. A target matching a local normal-reading sentence is a biography, and every other target is a movie review. This is checked against GLIM's task labels where a sentence appears in both.
3. **Protected data:** targets matching v1 test sentences, v2 test blocks or their buffers are removed by text.
4. **Score:** expected 10-way with all-MiniLM-L6-v2, in two pools: all scored sentences (mixed), and each sentence against its own corpus only (within corpus).
5. **Controls:** each model's noise-input outputs, and a predictor that knows only the corpus label.

## Decision, per model

Each comparison needs a gap of at least 2 points and a 95% lower bound above 0, from resampling sentences.

| Result | Label |
|---|---|
| Mixed pool does not beat chance and noise | No retrieval above chance or noise in these outputs |
| Mixed pool beats both, and the within-corpus score does not | The mixed score comes mostly from telling the corpora apart |
| Mixed pool beats both, and so does the within-corpus score | The score holds within a corpus |

## Counts, per model

Each file holds 1,407 rows and 104 sentences: 63 labelled biography and 41 movie review. 25 protected biography sentences are removed, which leaves 1,072 rows and 79 sentences (38 biographies and 41 movie reviews).

## Limits known in advance

- **Labels by elimination:** a biography sentence missing from the local lists would be counted as a movie review. The run stops if any label disagrees with GLIM's.
- **No reader labels:** intervals come from sentences only.
- **Teacher-forced outputs are secondary:** they are conditioned on the target text, so their scores say little about EEG.
- **T5 file names are undocumented:** `EEG_noise`, `noise_EEG` and `noise_noise` are reported as named.
- **Task-specific reading is left out:** the four `task1_task2_task3` files cannot be labelled with local lists.
- **Unknown checkpoints:** it is not known which runs produced these files.

## Synthetic checks before freezing

Over 20 seeds each, random outputs, corpus-only outputs and sentence-level outputs earned the first, second and third labels on 20 of 20.

## Changes made after approval and before freezing

No score had been computed when these were made. The files were parsed and counted only.

- **Decision rule:** the draft gave the second label when the mixed score beat the within-corpus score. A model with real sentence-level signal would also do that, because other-corpus distractors are easier. The second label now needs the within-corpus score to fail against chance or noise.
- **Pools:** the within-corpus score compares each sentence with its own corpus, as one pooled score, not as two separate pools.
- **T5 noise arm:** the label uses the `EEG_noise` file, read as trained on EEG and tested on noise. The other two noise files are secondary.
- **Commits:** the draft gave the wrong commit for the NeuSpeech repository. Its only branch is `main`, at `0158240`, and `master` in its links resolves there. `a07cf91` is the first repository's commit.
- **Label check is weak:** only 6 of the 104 sentences appear in GLIM's files. All 6 agree. The counts fit the expected split of about 40 movie reviews and 65 biographies.
- **Protected sentences** are matched by near-copies as well as exact text.
- **Seen before scoring:** the BART plain file holds only 4 distinct outputs across its 1,407 rows.

## Stop rule

Stop after this run, whatever the outcome. No extra files, embeddings or settings under this plan. No final test data is scored.
