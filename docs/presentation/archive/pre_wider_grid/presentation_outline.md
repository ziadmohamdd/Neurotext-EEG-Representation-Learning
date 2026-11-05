# NeuroText presentation outline

Draft for review; transfer remains pending identity evidence. Build slides from
the executed results in [the report](research_results/REPORT.md). Avoid claiming
successful semantic decoding or competitive published-paper accuracy.

An editable [14-slide draft](presentation/NeuroText_draft.pptx) expands this
outline with the archived charts and speaker notes. Review it before presenting.

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

Both transfer directions have executable software paths, but strict experiments
remain unexecuted until cross-release identity evidence is verified. Mention the
maintainer inquiry as an evidence request, not a completed identity audit.

Conclusion: these tested representations do not demonstrate reliable sentence
retrieval beyond matched controls. The contribution is a reproducible pipeline,
transparent controls and documented negative results. Remaining work and failed
or unexecuted requirements must remain visible.

## Backup material

Keep original configs, splits/checkpoints, 79 exported reports (including
repeated evaluations), cohort/exclusion audits, candidate hashes and the software
validation report available. Explain that report counts are not independent
methods and that synthetic tests verify software, not scientific performance.
