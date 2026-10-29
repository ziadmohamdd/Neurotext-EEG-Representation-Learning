# NeuroText and published EEG-to-language work

Protocol review, 2026-10-29. This is a source-verified methodological comparison,
not an independent reproduction of published numerical results. The reviewed
primary repository documents and code are pinned and SHA256-hashed in
[literature_sources.json](research_results/literature_sources.json).

## Open-vocabulary EEG-to-text decoding

Zhenhailong Wang and Heng Ji, *Open Vocabulary Electroencephalography-to-Text
Decoding and Zero-Shot Sentiment Classification*, AAAI 2022, 36(5), 5350–5358.
The citation and article link are provided by the
[official implementation](https://github.com/MikeWangWZHL/EEG-To-Text/blob/a07cf91452a4ee67eb413c38ff7bab56c6987545/README.md).

The original repository carries an important notice dated 2024-06-29:
"The original code has an unintential bug in evaluation that causes an inaccurate
assessment of the model’s true capability." It directs readers to the
[NeuSpeech correction repository](https://github.com/NeuSpeech/EEG-To-Text/blob/01582402809aa4fd8079dfb397faf04babc5d9e8/README.md)
and the analysis at https://arxiv.org/abs/2405.06459 . This warning comes from
the original authors' repository; it is not a new accusation based on NeuroText's
scores. The correction repository describes implicit teacher forcing during
evaluation and uses `model.generate` for the revised predictions.

| Aspect | Reviewed EEG-to-text implementation/correction | NeuroText executed experiments |
|---|---|---|
| Prediction task | Generate text with pretrained language-model decoders | Rank a fixed pool of sentence embeddings |
| EEG input | Word-level fixation-related features; default eight bands across 105 electrodes (840 values per word) | Sentence-aligned time-domain waveforms, plus statistics or learned features |
| Reading conditions | Configurations can combine SR, NR, TSR and ZuCo2 NR | NR-only cohorts, evaluated by version |
| Split | Correction evaluation sets `unique_sent`; its `ALL`-subject loader partitions sentence indices 80/10/10 within each subject | Subjects held out; separate strict subject-and-content split |
| Language component | BART and other large generation models in inspected code | Cached MiniLM sentence targets; ridge or contrastive projection |
| Metrics | BLEU, SacreBLEU, ROUGE, WER and CER in correction evaluation | Recall@1/5/10 and MRR with explicit candidate pools |
| Evaluation caveat | Original repository explicitly acknowledges an evaluation bug | Retrieval is not decoder teacher-forced generation; original signal/timing and cohort audits still have limitations |

The split description above concerns the specific inspected correction-code
path, not every experiment in the original paper or every later model. The
loader also contains other settings. Likewise, seeing `generate` in code is not
a complete independent audit of every decoder path or dependency version.

**A BLEU score cannot be compared numerically with a retrieval accuracy or MRR.**
An all-subject sentence partition also differs from a held-out-subject study.
NeuroText cannot claim to outperform or reproduce this paper using its current
scores. A direct numerical comparison requires a matched task, representation,
stimuli, split, language-model access, decoding procedure and metric.

The correction makes negative controls especially relevant. Our Gaussian and
shuffled comparisons test whether real EEG contributes useful information under
our own protocol. Their near-chance or noise-equivalent outcomes are not proof
of the same failure mechanism in the published generator. Our standard-deviation-
matched Gaussian controls are also not automatically equivalent to another
study's noise construction.

## LaBraM foundation-model results

Wei-Bang Jiang, Li-Ming Zhao and Bao-Liang Lu, *Large Brain Model for Learning
Generic Representations with Tremendous EEG Data in BCI*, ICLR 2024.
[Official pinned source](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/README.md),
[paper](https://openreview.net/forum?id=QzTpTRVtrP).

The official description reports pretraining across roughly 2,500 hours and
around 20 EEG datasets and downstream tasks including abnormal/event
classification, emotion recognition and gait prediction. These are not the
same task or metric as ZuCo fixed-pool sentence retrieval.

The documented input recipe includes 0.1–75 Hz filtering, a 50 Hz notch,
200 Hz sampling, microvolt units and correct channel positional information.
Our implementation verifies the official weights and uses 200 Hz physical
inputs. However, it preserves existing ZuCo preprocessing, uses a documented
approximate 19-electrode positional adaptation and freezes the backbone. It
does not reproduce the paper's end-to-end downstream fine-tuning experiments.

Our valid internal comparison is therefore narrow: frozen LaBraM features
versus statistical/spectral features, with ridge heads on the same adapted
recordings and subject splits. LaBraM did not improve top-1 or MRR on either
reported split; some ZuCo1 higher-K metrics improved. This is evidence about
our adaptation, not a contradiction of the paper's classification results.

## Wording suitable for the report

> Our study evaluates fixed-pool sentence retrieval from sentence-aligned EEG
> under subject-held-out and subject-and-content-held-out protocols. Published
> EEG-to-text generation scores are not directly comparable because the tasks,
> input representations, language models, evaluation procedures and metrics
> differ. The original AAAI 2022 implementation also carries a later evaluation
> correction notice. Across our tested configurations, real EEG did not
> consistently outperform separately trained negative controls. We therefore
> do not claim competitive published-paper decoding performance.

## Scope still open

The literature/protocol discussion is supported by inspected primary sources.
A protocol-matched **numerical reproduction** remains unexecuted. It is required
before claiming paper-level competitiveness, not something that the present
qualitative comparison establishes. Implementing a new text generator would be
a substantial additional experiment, with separate data/model/hardware needs.

Neither this literature review nor the corrected generation repository resolves
whether the ZuCo1 and ZuCo2 recordings used here share participants. Strict
cross-dataset transfer remains blocked on that identity evidence. It must not
be relabeled as verified subject-independent transfer merely to fill a table.
