# Methodology

## Research question and representations

Test whether EEG while reading contains sentence-semantic information that transfers across participants and ZuCo versions. Compare (1) statistical/spectral features with train-standardized ridge, (2) a compact supervised EEG CNN, (3) the same encoder initialized with ZuCo self-supervision, and (4) frozen official LaBraM features with a trainable projection (ridge is also supported). No unrestricted text generation is implemented or claimed.

The compact encoder mixes channels with a 1x1 convolution, uses depthwise temporal convolution plus another temporal layer, GroupNorm and adaptive pooling, and returns 128/256-dimensional embeddings. Processing each true sequence separately prevents padding from changing convolutions or normalization. Input duration varies; deterministic center cropping to at most 1600 samples bounds local memory. Apply the identical crop rule to all comparisons, and disclose lost context.

SSL masks randomly chosen contiguous 20-sample blocks and minimizes MSE only at hidden samples. For observations with 2–20 valid samples, the block width is reduced to half the valid length (rounded down), retaining at least one visible block and one hidden block. Padding never enters the target; observations are not discarded by this rule. Zeroed regions must be reconstructed through a temporal context encoder, without a decoder skip path. This is a lightweight signal objective for small batches, not semantic supervision; interpolation/low-frequency bias is a limitation. Validation masks are held reproducible. Strict source training subjects alone are used in SSL; validation estimates the objective, and test subjects never enter optimization. A matching split digest and normalizer are mandatory before SSL initialization of alignment.

## Preprocessing

Internal waveform units are microvolts. Harmonize ordered shared electrodes and sampling rates before combining datasets. Default new-data recipe is 0.5–45 Hz fourth-order zero-phase bandpass, 50 Hz notch where below Nyquist, then polyphase anti-alias resampling to 200 Hz. Short segments failing filter padding are reported, not padded into fake EEG. A separate LaBraM recipe follows its 0.1–75 Hz, 50 Hz, 200 Hz contract. Already processed signals must not be filtered twice; repeated identical preprocessing reuses the previous record, changed preprocessing requires original imported inputs. Zero-phase filtering is applied within each sentence, never across subject/split boundaries; inspect boundary artifacts and consider continuous-record preprocessing before segmentation as a follow-up with real data.

Reject nonfinite signals. Default amplitude-artifact handling rejects observations exceeding 200 microvolts after filtering; flag/clip are explicit alternatives. This fixed threshold is an initial physiological heuristic requiring real-data diagnostics, not fitted quality ground truth. Diagnostics include finite fraction, RMS, peak, flat channels, duration, traces and PSD. No automatic ICA is fitted with test subjects.

Streaming channel normalization estimates population means/variances on permitted training signals only. The saved state is applied to validation/test. Ridge also fits a training-only feature scaler. LaBraM consumes physical units divided by 100, without signal z-scoring, and cached features receive train-only feature normalization.

Quality-aware models attenuate channels according to per-observation peak/standard-deviation transient ratio, capped at 1. This is label-free and fits no test-distribution statistics. It is a defined quality-aware ablation, not a proven denoiser. Compare real/quality-aware encoders on identical observations and severity levels.

## Semantic alignment and retrieval

Precompute normalized sentence embeddings on CPU with Sentence Transformers `all-MiniLM-L6-v2`. Save model name and resolved revision; use an explicit `text_revision` and keep the cache immutable for reproducibility. Free the text encoder before EEG optimization. Sentence identity uses NFKC, case folding and collapsed whitespace; this deliberately merges formatting/case variants but not paraphrases. Each candidate is a unique normalized sentence; repeated EEG observations remain separate queries.

EEG projection uses cosine-normalized vectors and symmetric multi-positive contrastive alignment. Repeated sentences in a batch are positives, never false negatives. Microbatches must contain at least two unique sentences; a rare singleton-class tail repeats an anchor from the same split. Gradient accumulation changes effective optimizer batch size but does **not** create additional contrastive negatives. This is a limitation of batch 2. Use larger Colab batches after profiling.

The candidate pool is fixed to unique sentences in the saved test split. Compare top-1, top-5, Recall@1/5/10 and MRR against exact random-ranking chance for that pool size. Ties use average ranks, a conservative convention rather than optimistic tie ordering. Store candidate keys and a pool hash; do not compare runs with differing pools as equivalent. The text encoder can embed test candidate text without fitting EEG parameters; for held-out-content claims, the source EEG/text pairs and all fitted transformations remain content-disjoint.

## Protocols and controls

Primary held-out-subject split assigns approximately 60/20/20 percent of subject identities to train/validation/test with deterministic seeds and explicit overlap checks. `subject_sentence` also partitions sentence hashes and drops observations inconsistent with their assigned subject/content split. This reduces data and may yield empty splits; fail rather than relaxing leakage checks. Optional LOSO is implemented as a reusable fold generator with one validation subject per fold.

Run each version separately for within-dataset comparisons. Cross-dataset transfer trains/selects on the source version and tests exclusively on the target, requiring shared channel order, sample rate, matching task set and audited participant identities. Report sentence overlap. This is zero-shot transfer, not target-domain adaptation. The library supports excluding source-overlapping target sentences through `transfer_split(..., unseen_sentences=True)`; report if none remain. Target-domain adaptation is not implemented or mislabeled.

Use separate Gaussian-input and shuffled-pair training runs, plus evaluation noise/label-permutation diagnostics. Gaussian noise is zero-mean and matched to the raw observation's per-channel standard deviation; no sentence content enters the input. Shuffling deranges sentence classes, retaining duplicate-observation consistency. All corruption is applied in physical signal space before training-fitted normalization. Per-observation deterministic seeds align corruption across model comparisons. The default robustness suite changes only held-out input: additive noise ratios 0/.25/.5/1, channel dropout 0/.1/.25/.5 and temporal masking 0/.1/.25/.5. No degraded test sample is used for training.

## Statistics and interpretation

Report each subject separately, micro query metrics and equal-subject macro means. Confidence intervals use 2000 subject-cluster bootstrap replicates, not independent sentence trials. With fewer than two held-out subjects, return no CI; use LOSO/repeated splits for credible variability estimates. Default repeated seeds are 42,43,44. Repeated splits are not independent new participants. Small fixed candidate pools, repeated stimulus content, quality filtering, montage adaptation, and text embedding biases limit interpretation. Do not infer linguistic brain decoding from chance-level or noise-control-equivalent results.
