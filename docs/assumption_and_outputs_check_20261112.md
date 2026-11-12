# Two checks for the audit paper: list order and other public outputs

12 November 2026. Reading only. Nothing was downloaded, trained or scored, and no EEG file was opened. Paper details come from automated page summaries, so read each source before citing it.

## 1. Does sentence order match presentation order?

**Finding: both dataset papers say every reader saw the sentences in the same fixed order.** That supports the block design. It does not directly confirm that the sentence IDs in the files follow that order.

| Release | What the paper says | Source |
|---|---|---|
| ZuCo 1.0 | "The sentences were presented to all subjects in identical order." Normal reading came first in session one. The eye tracker was recalibrated in blocks of 60 sentences | [Hollenstein et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289117/) |
| ZuCo 2.0 | "The order of blocks and sentences within blocks was identical for all subjects." 14 blocks of about 50 sentences, alternating normal and task-specific reading, in one session | [Hollenstein et al. 2019](https://ar5iv.labs.arxiv.org/html/1912.00903), Section 3.3 |

**What this means for our tests:**

- **Fixed order is confirmed.** Position effects are therefore shared by all readers, which is why averaging across readers does not remove them.
- **ID order is still inferred.** The evidence is indirect: EEG predicts sentence ID at a correlation of 0.80 and 0.88, which would be unlikely if IDs were unrelated to time. Local file names fit about 50 sentences per recording file (up to `NR6` for ZuCo 1.0 and `NR7` for ZuCo 2.0).
- **A new caveat for ZuCo 2.0.** Its normal-reading list is interrupted about every 50 sentences by a task-specific block and a break. Our held-out blocks are 25 sentences with 10-sentence buffers, so they do not line up with these recording blocks. Training sentences can share a recording block with held-out ones. This would favour EEG, and EEG still did not beat noise, so the null results stand. A stricter split would hold out whole recording blocks.
- **A direct check is possible.** The OSF task materials list the sentences per block. Those files are small, but they are a download and need the owner's approval with exact sizes first.

## 2. Is any reader in both releases?

**Still unresolved.** Neither paper says whether the 18 ZuCo 2.0 readers include any of the 12 ZuCo 1.0 readers. The owner emailed the dataset's first author on 29 October 2026 and has had no reply. Cross-release results stay labelled exploratory.

## 3. Other models with public outputs

**Finding: two more models publish generated test text, in three sets of files.** Each file mixes movie reviews and biographies, so the GLIM check could be repeated on them without waiting for SemKey.

| Model | Files | What they hold |
|---|---|---|
| BrainTranslator (BART), Wang and Ji 2022 | [MikeWangWZHL/EEG-To-Text `results/`](https://github.com/MikeWangWZHL/EEG-To-Text/tree/main/results), 5 files listed | Predicted and true text per test row, with and without teacher forcing, and with noise input |
| T5 version, Jo et al. 2024 | [NeuSpeech/EEG-To-Text `results/` on `master`](https://github.com/NeuSpeech/EEG-To-Text/tree/master/results), 8 files listed | Target, prediction with teacher forcing and free prediction, for combinations of EEG and noise input |

One file from each repository was read to see its format. They are plain text of about 0.4 MB and 0.6 MB.

**Limits to check before using them:**

- **No task or reader labels.** The task would have to be inferred by matching each target to our sentence lists.
- **Weak outputs.** Without teacher forcing, the BART model repeats one sentence for many rows. [Jo et al.](https://arxiv.org/abs/2405.06459) already show these models score the same on noise. So the likely result is chance-level retrieval everywhere, which adds a second model but a less striking one than GLIM.
- **File name meanings are undocumented.** `EEG_noise`, `noise_EEG` and `noise_noise` probably mean training and test input, but the README does not say.
- **No licence stated** in the NeuSpeech repository. The files also contain ZuCo sentence text, so they should stay out of this repository.
- **Not exhaustive.** Two searches and two repositories were checked. DeWave, BELT-2, CET-MAE and R1 Translator were not confirmed either way.

## Suggested next steps

1. Decide whether to repeat the GLIM check on these outputs under a frozen plan. It needs about 13 small text files and no EEG.
2. Decide whether to download the OSF task materials to check list order directly.
3. Send one reminder about reader overlap.
