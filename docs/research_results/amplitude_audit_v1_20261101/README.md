# Full-width relative TRT and target-head audit — training readers only

**These are selected inner-training-reader cross-validation results, not outer validation or test accuracy.** No outer validation/test array was extracted, no raw source MAT file was reopened, and no previous model or report was replaced. The fixed run has ended.

## Diagnosed hypotheses

The prior word-order pilot retained 75.2488% of standardized training-word variance in 16 PCA components. Variance retention does not measure retained semantic information and does not prove PCA caused its failure. No confirmed pairing/ranking implementation bug was identified in the reviewed paths; earlier saved-model verification remains intact.

Original training-reader mean feature norms range from 23.5279 to 83.5290 (ratio 3.5502). The new transform gives 6.8578 to 10.3459 (ratio 1.5086). This is an amplitude diagnostic, not a demonstration of successful participant alignment or retrieval.

The new representation uses all 840 supplied TRT coordinates (eight bands × 105 source channels), with no PCA. Complete words undergo signed log1p and subtraction of the 105-channel mean separately within each band. For the observed nonnegative real values this is log1p; negative Gaussian draws use the same signed transform without clipping. Sentence features average the transformed complete words. All-missing ZMG/70 remains in training, handled by training-fitted imputation. This changes the representation and drops word order; it is not an isolated PCA-component ablation.

## Frozen design and selection

Only the original eight training readers and 2,245 observations enter this run. The fixed candidate vocabulary consists of their 300 unique sentence classes. Each fold holds one training reader out. Source word-cache arrays read are exactly `train_words` and `train_offsets`; structural input reads only `count_duration_only_train`. ZJS/ZKW outer validation and ZAB/ZKB test EEG are not used. Metadata manifests contain identities for provenance but only declared training rows are selected.

Each fold fits Gaussian moments from the other seven readers' words, draws null words with original word counts/masks, then applies exactly the same relative transform. Gaussian values retain source-scale marginal moments before the nonlinear transform, not after it. A Gaussian null does not preserve the full positive real-feature distribution or covariance; superiority over it alone would not establish EEG semantics.

Structural inputs are the prior 14 count/duration summaries plus a count basis rebuilt from fold-training counts. The structural median/standard transform is frozen and shared across structure-bearing conditions within a fold. Real and Gaussian median imputation/standardization fit fold-training rows only. EEG weights apply after scaling. Durations are source sample counts at 500 Hz, not inferred onset times.

Two heads use the same ridge design: (1) semantic sentence embeddings with cosine retrieval, and (2) 300 one-hot class targets ranked by raw class score. The latter is **closed-set classification of seen sentence classes**, not unrestricted text generation, unseen-text generalization, or proof of semantic understanding. Sentence IDs/word text are not numeric EEG inputs; class labels are supervision only. Gaze/stimulus-assisted source segmentation and missingness remain annotation information.

Each head/combined arm has the same nine settings: alpha [1,10,100] × post-scaling EEG weight [0,0.1,1]. Structural-only and amplitude-only comparators have three alphas each. Selection maximizes equal-reader mean Recall@1; roundoff ties prefer zero/smaller EEG weight, then larger alpha. Combined arms have more search opportunity than structural-only; real and Gaussian search budgets match. Zero EEG weight exactly reuses structural predictions. The class-shuffled arm consistently deranges the entire training target mapping, including the structural association; it is a whole-pair control, not an isolated EEG-block shuffle.

There are 576 scored fold/settings/head rows (36 settings per head × 2 heads × 8 readers). All settings are archived. Selected CV scores reuse the same eight readers for choice and description, so are optimistic exploration, not independent confirmation or significance tests. The prior studies also used these training participants. Results below must not be directly compared with historical 512-query/298-candidate validation accuracy.

## Selected training-reader CV results

Recall is equal-reader macro percent; MRR is not accuracy. Chance Recall@1 for 300 candidates is 0.3333%.

| Head | Condition | Alpha | EEG weight | Macro R@1 % | Macro R@5 % | Macro R@10 % | Macro MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| semantic_embedding | structural_only | 1 | 0 | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| semantic_embedding | eeg_only | 100 | 1 | 0.3547 | 1.3798 | 2.7290 | 0.020519 |
| semantic_embedding | gaussian_only | 100 | 1 | 0.6959 | 2.3128 | 4.2199 | 0.025876 |
| semantic_embedding | structural_real | 1 | 0 | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| semantic_embedding | structural_gaussian | 1 | 0 | 16.1129 | 40.2867 | 52.0525 | 0.280271 |
| semantic_embedding | structural_shuffled | 100 | 1 | 0.3603 | 2.1483 | 3.8220 | 0.021886 |
| known_sentence_class | structural_only | 10 | 0 | 20.9088 | 63.4175 | 89.3180 | 0.400506 |
| known_sentence_class | eeg_only | 1 | 1 | 0.3575 | 1.1614 | 2.6305 | 0.019292 |
| known_sentence_class | gaussian_only | 10 | 1 | 0.3954 | 1.7278 | 3.8723 | 0.022132 |
| known_sentence_class | structural_real | 10 | 0 | 20.9088 | 63.4175 | 89.3180 | 0.400506 |
| known_sentence_class | structural_gaussian | 10 | 0 | 20.9088 | 63.4175 | 89.3180 | 0.400506 |
| known_sentence_class | structural_shuffled | 1 | 1 | 0.5866 | 1.7994 | 3.0579 | 0.021740 |

## Prespecified next-stage screen

Before the run, a head's structural+real model had to exceed both structural-only and structural+Gaussian by at least two percentage points of macro Recall@1 and show positive reader differences to both on at least six of eight readers. This is a descriptive screen, not a significance test. The run stops without any outer evaluation regardless.

| Head | Real − structural R@1 pp | Real − Gaussian-combined R@1 pp | Positive readers vs structural | Positive readers vs Gaussian | Passed |
|---|---:|---:|---:|---:|---|
| semantic_embedding | +0.0000 | +0.0000 | 0/8 | 0/8 | False |
| known_sentence_class | +0.0000 | +0.0000 | 0/8 | 0/8 | False |

**Neither head passed.** Reducing reader amplitude differences and retaining all channels did not establish a sufficiently consistent incremental EEG advantage. No additional outer validation or test evaluation was justified or performed.

## Reader-level selected results

| Head | Condition | Training holdout reader | Queries | R@1 % | MRR |
|---|---|---|---:|---:|---:|
| semantic_embedding | structural_only | ZDM | 289 | 14.8789 | 0.270690 |
| semantic_embedding | structural_only | ZDN | 297 | 12.7946 | 0.254211 |
| semantic_embedding | structural_only | ZGW | 281 | 18.1495 | 0.290449 |
| semantic_embedding | structural_only | ZJM | 257 | 15.5642 | 0.275772 |
| semantic_embedding | structural_only | ZJN | 294 | 17.3469 | 0.277953 |
| semantic_embedding | structural_only | ZKH | 294 | 16.3265 | 0.285932 |
| semantic_embedding | structural_only | ZMG | 286 | 16.4336 | 0.285010 |
| semantic_embedding | structural_only | ZPH | 247 | 17.4089 | 0.302149 |
| semantic_embedding | eeg_only | ZDM | 289 | 0.0000 | 0.019047 |
| semantic_embedding | eeg_only | ZDN | 297 | 0.6734 | 0.023793 |
| semantic_embedding | eeg_only | ZGW | 281 | 0.3559 | 0.024616 |
| semantic_embedding | eeg_only | ZJM | 257 | 0.7782 | 0.025100 |
| semantic_embedding | eeg_only | ZJN | 294 | 0.0000 | 0.016541 |
| semantic_embedding | eeg_only | ZKH | 294 | 0.6803 | 0.021427 |
| semantic_embedding | eeg_only | ZMG | 286 | 0.3497 | 0.016604 |
| semantic_embedding | eeg_only | ZPH | 247 | 0.0000 | 0.017025 |
| semantic_embedding | gaussian_only | ZDM | 289 | 0.3460 | 0.021266 |
| semantic_embedding | gaussian_only | ZDN | 297 | 1.3468 | 0.027251 |
| semantic_embedding | gaussian_only | ZGW | 281 | 0.0000 | 0.019876 |
| semantic_embedding | gaussian_only | ZJM | 257 | 0.3891 | 0.022946 |
| semantic_embedding | gaussian_only | ZJN | 294 | 1.7007 | 0.034417 |
| semantic_embedding | gaussian_only | ZKH | 294 | 0.6803 | 0.024956 |
| semantic_embedding | gaussian_only | ZMG | 286 | 0.6993 | 0.026883 |
| semantic_embedding | gaussian_only | ZPH | 247 | 0.4049 | 0.029412 |
| semantic_embedding | structural_real | ZDM | 289 | 14.8789 | 0.270690 |
| semantic_embedding | structural_real | ZDN | 297 | 12.7946 | 0.254211 |
| semantic_embedding | structural_real | ZGW | 281 | 18.1495 | 0.290449 |
| semantic_embedding | structural_real | ZJM | 257 | 15.5642 | 0.275772 |
| semantic_embedding | structural_real | ZJN | 294 | 17.3469 | 0.277953 |
| semantic_embedding | structural_real | ZKH | 294 | 16.3265 | 0.285932 |
| semantic_embedding | structural_real | ZMG | 286 | 16.4336 | 0.285010 |
| semantic_embedding | structural_real | ZPH | 247 | 17.4089 | 0.302149 |
| semantic_embedding | structural_gaussian | ZDM | 289 | 14.8789 | 0.270690 |
| semantic_embedding | structural_gaussian | ZDN | 297 | 12.7946 | 0.254211 |
| semantic_embedding | structural_gaussian | ZGW | 281 | 18.1495 | 0.290449 |
| semantic_embedding | structural_gaussian | ZJM | 257 | 15.5642 | 0.275772 |
| semantic_embedding | structural_gaussian | ZJN | 294 | 17.3469 | 0.277953 |
| semantic_embedding | structural_gaussian | ZKH | 294 | 16.3265 | 0.285932 |
| semantic_embedding | structural_gaussian | ZMG | 286 | 16.4336 | 0.285010 |
| semantic_embedding | structural_gaussian | ZPH | 247 | 17.4089 | 0.302149 |
| semantic_embedding | structural_shuffled | ZDM | 289 | 0.3460 | 0.023558 |
| semantic_embedding | structural_shuffled | ZDN | 297 | 0.6734 | 0.024726 |
| semantic_embedding | structural_shuffled | ZGW | 281 | 0.0000 | 0.016878 |
| semantic_embedding | structural_shuffled | ZJM | 257 | 0.7782 | 0.029051 |
| semantic_embedding | structural_shuffled | ZJN | 294 | 0.3401 | 0.019667 |
| semantic_embedding | structural_shuffled | ZKH | 294 | 0.3401 | 0.021756 |
| semantic_embedding | structural_shuffled | ZMG | 286 | 0.0000 | 0.014777 |
| semantic_embedding | structural_shuffled | ZPH | 247 | 0.4049 | 0.024673 |
| known_sentence_class | structural_only | ZDM | 289 | 19.3772 | 0.386117 |
| known_sentence_class | structural_only | ZDN | 297 | 17.8451 | 0.365721 |
| known_sentence_class | structural_only | ZGW | 281 | 17.4377 | 0.382487 |
| known_sentence_class | structural_only | ZJM | 257 | 20.2335 | 0.413858 |
| known_sentence_class | structural_only | ZJN | 294 | 21.0884 | 0.394179 |
| known_sentence_class | structural_only | ZKH | 294 | 23.8095 | 0.413082 |
| known_sentence_class | structural_only | ZMG | 286 | 22.3776 | 0.410269 |
| known_sentence_class | structural_only | ZPH | 247 | 25.1012 | 0.438330 |
| known_sentence_class | eeg_only | ZDM | 289 | 0.3460 | 0.021007 |
| known_sentence_class | eeg_only | ZDN | 297 | 0.0000 | 0.016524 |
| known_sentence_class | eeg_only | ZGW | 281 | 0.0000 | 0.015711 |
| known_sentence_class | eeg_only | ZJM | 257 | 0.3891 | 0.020369 |
| known_sentence_class | eeg_only | ZJN | 294 | 0.3401 | 0.020299 |
| known_sentence_class | eeg_only | ZKH | 294 | 0.6803 | 0.021941 |
| known_sentence_class | eeg_only | ZMG | 286 | 0.6993 | 0.018744 |
| known_sentence_class | eeg_only | ZPH | 247 | 0.4049 | 0.019738 |
| known_sentence_class | gaussian_only | ZDM | 289 | 0.3460 | 0.023498 |
| known_sentence_class | gaussian_only | ZDN | 297 | 0.3367 | 0.021962 |
| known_sentence_class | gaussian_only | ZGW | 281 | 0.3559 | 0.017789 |
| known_sentence_class | gaussian_only | ZJM | 257 | 0.0000 | 0.019055 |
| known_sentence_class | gaussian_only | ZJN | 294 | 0.0000 | 0.018576 |
| known_sentence_class | gaussian_only | ZKH | 294 | 1.0204 | 0.028223 |
| known_sentence_class | gaussian_only | ZMG | 286 | 0.6993 | 0.027624 |
| known_sentence_class | gaussian_only | ZPH | 247 | 0.4049 | 0.020328 |
| known_sentence_class | structural_real | ZDM | 289 | 19.3772 | 0.386117 |
| known_sentence_class | structural_real | ZDN | 297 | 17.8451 | 0.365721 |
| known_sentence_class | structural_real | ZGW | 281 | 17.4377 | 0.382487 |
| known_sentence_class | structural_real | ZJM | 257 | 20.2335 | 0.413858 |
| known_sentence_class | structural_real | ZJN | 294 | 21.0884 | 0.394179 |
| known_sentence_class | structural_real | ZKH | 294 | 23.8095 | 0.413082 |
| known_sentence_class | structural_real | ZMG | 286 | 22.3776 | 0.410269 |
| known_sentence_class | structural_real | ZPH | 247 | 25.1012 | 0.438330 |
| known_sentence_class | structural_gaussian | ZDM | 289 | 19.3772 | 0.386117 |
| known_sentence_class | structural_gaussian | ZDN | 297 | 17.8451 | 0.365721 |
| known_sentence_class | structural_gaussian | ZGW | 281 | 17.4377 | 0.382487 |
| known_sentence_class | structural_gaussian | ZJM | 257 | 20.2335 | 0.413858 |
| known_sentence_class | structural_gaussian | ZJN | 294 | 21.0884 | 0.394179 |
| known_sentence_class | structural_gaussian | ZKH | 294 | 23.8095 | 0.413082 |
| known_sentence_class | structural_gaussian | ZMG | 286 | 22.3776 | 0.410269 |
| known_sentence_class | structural_gaussian | ZPH | 247 | 25.1012 | 0.438330 |
| known_sentence_class | structural_shuffled | ZDM | 289 | 0.3460 | 0.017894 |
| known_sentence_class | structural_shuffled | ZDN | 297 | 0.3367 | 0.020698 |
| known_sentence_class | structural_shuffled | ZGW | 281 | 0.3559 | 0.017952 |
| known_sentence_class | structural_shuffled | ZJM | 257 | 0.3891 | 0.022155 |
| known_sentence_class | structural_shuffled | ZJN | 294 | 1.0204 | 0.024086 |
| known_sentence_class | structural_shuffled | ZKH | 294 | 0.6803 | 0.023273 |
| known_sentence_class | structural_shuffled | ZMG | 286 | 0.3497 | 0.020125 |
| known_sentence_class | structural_shuffled | ZPH | 247 | 1.2146 | 0.027741 |

## Preservation, verification and literature context

All 1498 prior protected files remained SHA256-identical, including previous results, reports, presentation, checkpoints and selected source caches. Eight focused tests passed for relative features, ties, selection and existing ridge-block invariants. No full-suite count is claimed for this run.

The stored plan, all fold partitions and class mappings, every scored setting with query ranks, chosen settings, training-only audit, decision and implementation snapshots are archived here. The original word source cache remains untouched. This run stores CV evidence, not a newly validated deployable model; no model is selected using outer data.

Related primary work studies word-level Conformer classification followed by sentence retrieval ([Zhou et al., 2024](https://arxiv.org/abs/2408.04679)); this diagnostic is not a reproduction of its architecture, task or numerical results. Noise-controlled evaluation is motivated by [Jo et al., 2024](https://arxiv.org/abs/2405.06459), which found noise-equivalent performance for the EEG-to-text generators they evaluated; that result does not prove the same cause here.
