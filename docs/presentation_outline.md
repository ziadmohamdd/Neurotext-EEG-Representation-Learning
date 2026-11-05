# NeuroText presentation outline

Draft for review; exploratory transfer is executed while strict participant
independence remains unresolved. Build slides from
the executed results in [the report](research_results/REPORT.md). Avoid claiming
successful semantic decoding or competitive published-paper accuracy.

An editable [35-slide draft](presentation/NeuroText_consolidated_v2_20261105.pptx) expands this
outline with the archived charts and speaker notes. Review it before presenting.
The earlier [26-slide deck](presentation/NeuroText_consolidated.pptx) is unchanged.

The latest six slides concern **validation-only** spectral and incremental
results. The same seed-42 validation cohort is reused. Further tuning and test
evaluation are stopped; historical experimental steps below are context only.
Sections 11–13 describe slides 21–26. Slides 27–35 add the later pilots.

## 1. Research question

Can learned EEG representations retrieve sentence meaning across unseen readers
and unseen sentence content, beyond noise and shuffled-pair controls?
Introduce fixed-pool retrieval and the limited local hardware (RTX 3050, 4 GB).

## 2. Data and quality audit

Use ZuCo1 and ZuCo2 normal reading, ordered 105 channels, resampled to 200 Hz.
Show preparation counts (3,356 and 5,911), then duration-QC counts (3,349 and
5,884). Explain documented missing slots and original unusually short segments.
Keep the separate 19-channel/full-second adaptation cohorts explicit.

## 3. Shared implementation

Show: lazy sentence signals → training-only normalization → representations →
cached MiniLM sentence targets → retrieval ranks. Distinguish statistical ridge,
supervised CNN, masked-reconstruction SSL and frozen official LaBraM.
Code is shared by local/Colab clients; do not claim Colab execution was measured.

## 4. Evaluation and controls

Explain subject splits, source/train-only fitting, validation checkpoint
selection, candidate pools and multiple EEG observations per sentence.
Show Gaussian and shuffled conditions as separately trained controls.
Gaussian SSL covers both pretraining and alignment; shuffled SSL reuses label-free
real-EEG pretraining. Retrieval is not unrestricted text generation.

## 5. CNN and SSL findings

Use [CNN controls](research_results/cnn_controls_mrr.png) and
[SSL controls](research_results/ssl_controls_mrr.png). Highlight overfitting
histories and the absence of consistent real-EEG superiority. Reconstruction
improvement alone did not imply semantic retrieval improvement.

## 6. Pretrained representation comparison

Use [matched features](research_results/pretrained_comparison.png) and
[LaBraM controls](research_results/labram_controls_mrr.png). Explain verified
official weights and the approximate template-based positional mapping.
Statistical features had higher top-1/MRR on the reported splits; the adaptation
and one split limit conclusions about LaBraM generally.

## 7. Unseen subject and sentence content

Use [three-seed results](research_results/unseen_content_mrr.png). Candidate pools
are 60/68, not 300/344; do not compare raw accuracy across those protocols.
Explain that real EEG does not consistently beat Gaussian controls and that
three repeated splits are not independent participant cohorts.

## 8. Robustness and signal quality

Use [quality comparison](research_results/quality_robustness_mrr.png). Results
concern ZuCo2 ridge, one split/corruption seed. The quality heuristic is lower
in MRR for eight of nine nonzero conditions. No general robust-decoding claim.

## 9. Published work and limitations

Summarize [the source-verified literature review](published_work_comparison.md):
generation/classification metrics differ from retrieval; the original AAAI 2022
implementation documents an evaluation correction. No matched numerical paper
reproduction or performance-ranking claim is made. Discuss montage/reference,
original timing, small subject cohorts and repeated test inspection limitations.

## 10. Transfer status and conclusion

Both directions have 18 wider-grid exploratory ridge evaluations across seeds
42/43/44 with real, Gaussian and shuffled controls. Keep the six original-grid
seed-42 comparisons separate. Show every seed, mean/sample SD and paired
real-minus-Gaussian differences. Source validation alone selects alpha. Target
people and queries recur at every seed, so repeated splits are not independent
cohorts. Fifty shared sentence texts remain in the target pools. Strict transfer
remains unexecuted pending identity evidence.

Forward mean MRR: real 0.018529, Gaussian 0.019346, shuffled 0.018247.
Reverse: 0.023592, 0.023142, 0.021216. Paired real-minus-Gaussian MRR mean ± SD:
−0.000817 ± 0.000831 forward and +0.000450 ± 0.002237 reverse. Reverse
differences change sign. SD is descriptive, not a confidence interval.

Conclusion: these tested representations do not demonstrate reliable sentence
retrieval beyond matched controls. The contribution is a reproducible pipeline,
transparent controls and documented negative results. Remaining work and failed
or unexecuted requirements must remain visible.

## 11. Validation-only spectral pilot (slides 21–22)

Use the unchanged ZuCo1 seed-42 cohort: 2,245 training observations, 512
validation queries and 298 candidates. Label every new score validation-only.
Show all six pilot conditions. Count/duration-only reaches MRR 0.292012 and
17.7734% Recall@1 without measured spectral amplitudes. This structural
retrieval performance is separate from neural information. Each pilot arm
selects from 28 settings on this reused validation split.

## 12. Validation-only incremental comparison (slides 23–25)

All three models share the exact 62 structural inputs and frozen training-fit
preprocessing. Each combined model searches 196 settings, including zero EEG
weight. Explain validation selection and the greater search opportunity than
structural-only's 28 settings. The selected MRR values are 0.292012 structural,
0.292310 structural+real and 0.292156 structural+Gaussian. Tiny real increments
of +0.000299 and +0.000154 accompany lower Recall@1 and opposite reader effects.
The structural baseline's appearance in both studies is not an independent
replication. Repeated validation reuse creates selection optimism.

## 13. Current conclusion and stopping status (slide 26)

The structural baseline performs well on the annotation-assisted task. These
configurations do not show a consistent useful incremental EEG advantage.
Every new score is validation-only. Stop further tuning and test evaluation;
no test features were extracted. Preserve prior artifacts and their separate
historical/test/exploratory-transfer protocol labels.

## 14. Later pilots and N-way (slides 27–33)

Label each scope on the slide. Ordered-word ridge uses reused outer validation.
Relative-amplitude, GRU, retrieval-loss and word-pretraining scores are
training-reader CV. Word-task scores are not sentence accuracy. N-way values are
replayed expectations with random distractors; smaller N changes difficulty and
chance, not the model. Structural scores are annotation-derived, not EEG decoding.

## 15. Brief coverage and current conclusion (slides 34–35)

Steps 1–4 are done. Quality analysis, strict transfer and degraded-condition
coverage remain partial. The result is a controlled negative result. It does not
show that EEG lacks language information. New experiments need a revised
hypothesis and a separately scoped task.

## Backup material

Keep original configs, splits/checkpoints, 79 exported reports (including
repeated evaluations), cohort/exclusion audits, candidate hashes and the software
validation report available. Explain that report counts are not independent
methods and that synthetic tests verify software, not scientific performance.
Include the wider-grid provenance archive. The deck's historical validation is 139 passed,
2 optional skips with `EEG_LABRAM_CACHE` unset (exact reasons in the report), and
0 deselected. No additional experiments run during consolidation.
