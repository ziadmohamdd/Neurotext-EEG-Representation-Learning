# Versioned TRT spectral-feature pilot — seed-42 validation only

**No useful validation advantage over the strongest control. Stopped without test extraction or evaluation.**

The pilot uses `zuco1_trt_sentence_mean_pilot_v1`, a new feature view with the original sentence targets and candidate embeddings. No previous data, checkpoint or experiment result was replaced. The punctuation repair was not applied. All six conditions use exactly the same 2,245 training and 512 validation observations and the same **298 unique validation candidates**. MRR remains reciprocal rank, not accuracy.

## Frozen design and feature provenance

Source inputs are the ten ZuCo1 training/validation NR MAT files; their hashes match the pinned download plan. ZAB/ZKB test MAT files and test voltage arrays were not opened by the pilot. Original source slot indices bind each observation; parent sentence content is checked against the unchanged target and never enters the numeric model matrix. **Word text is not decoded**, token/word IDs are not inputs, and no sentence/subject/session identity, pupil, bounding-box coordinates or raw eye-tracking trace enters a model. IDs are used solely for provenance, splitting, target lookup and deterministic Gaussian RNG.

The selected family is **TRT**, with band-major `t1,t2,a1,a2,b1,b2,g1,g2` and the existing ordered 105 channels. Each word supplies 840 source Hilbert-envelope amplitude values, retaining NaNs. No spectral differences or other fixation-measure EEG families are included. Each coordinate is averaged equally across its finite word values. This is an unweighted word mean over the provided TRT vectors, whose producer already averages available fixation-level envelope means. All-missing sentence coordinates stay NaN until training-only imputation. No duration weighting or spectral-to-waveform conversion occurs.

The [source audit](word_eeg_feasibility_20261030.md) documents band/channel definitions, source hashes and boundary limits. The code defines bands 4–6, 6.5–8, 8.5–10, 10.5–13, 13.5–18, 18.5–30, 30.5–40 and 40.5–49.5 Hz; the descriptor instead gives gamma2 40–49.5 Hz. No attempt was made to recompute or silently correct the provided features. Exact producer identity for every file is not proven. These are gaze-assisted, annotation-segmented features from the supplied session-level processing. They are not a strictly eight-second-window-matched or online EEG-only replacement. The sampled source boundary tail remains a limitation of the provided feature view.

**Duration units are stored sample counts at 500 Hz, not milliseconds.** The audited source code uses inclusive slices, with raw snippet `N=duration+1` in applicable examples. The count/duration control uses those sample counts as stored; it never invents EEG onsets or fills inter-fixation gaps.

## Matched conditions and prevention of target leakage

Every spectral arm has the same 840 amplitude positions plus three structural values (displayed word count, complete-TRT word count, complete fraction) and a training-derived one-hot word-count basis. The basis uses 47 training-observed counts and one unseen-count bucket: **891 features** in total. Count values and missingness are deliberate privileged non-neural information, not hidden as EEG. Providing the same structural block to real, Gaussian and zero conditions makes an EEG gain face a competitive structural control. The count basis is derived from actual word-record counts, never sentence IDs or labels. The 298 candidates contain eight sentence classes unique by word count; this known shortcut makes those controls essential.

| Condition | Numeric input and target rule |
|---|---|
| Real TRT | 840 amplitude means + shared structural/count block; correct training targets |
| Matched Gaussian | Independent Gaussian values at every finite word/coordinate, using only training-word marginal mean/population SD; exact original word counts, per-coordinate masks and shared structural block |
| Class-shuffled | The exact real input; a consistent derangement of the 300 training sentence classes, shared across repeated reader observations; correct validation targets |
| Count/duration-only | 14 count/duration summaries + the same 48-count basis (62 features), with no measured spectral amplitudes |
| Zero/missingness-only | Zero amplitude values at valid positions and the original missing positions, then the same pooling and structural/count block; no EEG amplitude information |
| Existing sentence ridge | The unchanged `eeg_features()` recipe: 105-channel mean/SD/range/RMS/absolute difference and five log-Welch-band features (1,050 total), from the original 200 Hz voltage arrays and center crop ≤1,600 samples |

Gaussian moments use **only the 32,848 complete training word observations**, not validation or test values, per-sentence real amplitudes or target embeddings. The source currently has joint all-band missingness per word; an unexpected partly missing word would fail the pilot rather than silently change the protocol. The Gaussian mask/shape was checked against real input for all 2,757 observations. Marginal matching does not preserve neural cross-channel/band covariance; negative Gaussian draws are feature-domain controls, not physically valid positive envelopes.

Count/duration summaries include word/fixated/complete/missing counts, missing fraction, fixation total/mean/SD and TRT total/mean/SD/min/median/max. Their source is gaze/structural metadata. The zero control retains shape/missingness information, rather than being an uninformative global zero vector. All-missing inputs are retained. Class shuffling affects training labels only; validation labels and candidate order stay fixed.

The sentence baseline reuses the saved **training-fitted** voltage normalizer for the identical split and refits its ridge model. Its feature recipe is unchanged; its new imputation/scaling/alpha search is the same as every pilot condition. This is not a reused score from another cohort. It also has a different signal window and lacks the word view’s annotation-derived structure; the real-minus-structural controls are the primary EEG-content assessment.

## Coverage and exclusions

| Split | Eligible/used observations | Source word records | Complete TRT words | All-missing observations retained |
|---|---:|---:|---:|---:|
| train | 2,245 | 48,865 | 32,848 | 1 |
| val | 512 | 11,203 | 7,101 | 0 |

There are **no new exclusions**, no candidate removal and no recovery of source slots excluded by the previous sentence-EEG preparation. ZMG/70 is the one all-missing training observation; its amplitudes are imputed using training data while its measured structure is retained in every condition. The original duration QC already removed ZDM/168, 173, 195; ZDN/278; ZJM/275; ZMG/71 (six training observations), and ZJS/250 (one validation observation). Those exclusions are inherited, not selected from this pilot’s scores. The original 592 test observations remain outside this validation-only feature cache. The earlier availability audit found six all-missing test cases; none was extracted, dropped or evaluated here.

## Training-only preprocessing and validation selection

For each condition, missing aggregate values are filled with training feature means or medians; an entirely missing training column has a recorded zero fallback. Standard scaling or robust median/IQR scaling is fitted on that condition’s imputed training matrix only. The word-count basis, Gaussian moments, voltage normalizer, ridge coefficients and intercept all use training data only. Validation is transformed with the frozen parameters; it never fits preprocessing.

The frozen grid is **2 imputation rules × 2 scaling rules × 7 alphas = 28 settings per condition** (168 total), with alphas `[0.1,1,10,100,1000,10000,100000]`. Float64 SVD ridge predicts the unchanged normalized 384-dimensional sentence embeddings. Maximum full-pool validation MRR selects the setting independently in each arm; ties within 1e-12 keep the first declared entry. All selection rows and fitted preprocessing parameters are saved. Recall values did not choose hyperparameters. Validation is reused across 28 settings and prior investigations, so the selected scores are optimistic model-selection measurements, not independent confirmatory results.

## Measured validation results

Recall is percent of 512 queries against the same 298 candidates.

| Condition | R@1 % | R@5 % | R@10 % | MRR | Imputation | Scaling | Alpha |
|---|---:|---:|---:|---:|---|---|---:|
| real_trt | 12.3047 | 32.2266 | 41.4062 | 0.226986 | median | standard | 100 |
| matched_gaussian | 10.7422 | 23.6328 | 28.9062 | 0.177208 | median | standard | 100 |
| class_shuffled | 0.3906 | 2.1484 | 3.7109 | 0.022676 | mean | robust_iqr | 0.1 |
| count_duration_only | 17.7734 | 40.6250 | 51.3672 | 0.292012 | median | standard | 10 |
| zero_missingness_only | 16.4062 | 40.2344 | 52.3438 | 0.281028 | mean | standard | 0.1 |
| existing_sentence_ridge | 0.5859 | 2.7344 | 4.1016 | 0.026320 | mean | standard | 1 |
| Uniform random ranking expectation | 0.3356 | 1.6779 | 3.3557 | 0.021060 | — | — | — |

### Real-minus-control and sentence-baseline differences

Recall differences are percentage points; MRR differences are absolute.

| Comparison | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |
|---|---:|---:|---:|---:|
| real_minus_matched_gaussian | +1.5625 | +8.5938 | +12.5000 | +0.049778 |
| real_minus_class_shuffled | +11.9141 | +30.0781 | +37.6953 | +0.204310 |
| real_minus_count_duration_only | -5.4688 | -8.3984 | -9.9609 | -0.065026 |
| real_minus_zero_missingness_only | -4.1016 | -8.0078 | -10.9375 | -0.054043 |
| real_minus_existing_sentence_ridge | +11.7188 | +29.4922 | +37.3047 | +0.200666 |

### Validation readers

| Condition | Reader | Queries | R@1 % | R@5 % | R@10 % | MRR |
|---|---|---:|---:|---:|---:|---:|
| real_trt | zuco1:ZJS | 216 | 11.5741 | 28.7037 | 38.8889 | 0.208341 |
| real_trt | zuco1:ZKW | 296 | 12.8378 | 34.7973 | 43.2432 | 0.240591 |
| matched_gaussian | zuco1:ZJS | 216 | 11.5741 | 21.7593 | 28.7037 | 0.174234 |
| matched_gaussian | zuco1:ZKW | 296 | 10.1351 | 25.0000 | 29.0541 | 0.179378 |
| class_shuffled | zuco1:ZJS | 216 | 0.0000 | 2.3148 | 3.7037 | 0.021469 |
| class_shuffled | zuco1:ZKW | 296 | 0.6757 | 2.0270 | 3.7162 | 0.023557 |
| count_duration_only | zuco1:ZJS | 216 | 17.5926 | 42.1296 | 50.9259 | 0.288718 |
| count_duration_only | zuco1:ZKW | 296 | 17.9054 | 39.5270 | 51.6892 | 0.294415 |
| zero_missingness_only | zuco1:ZJS | 216 | 17.5926 | 40.7407 | 52.7778 | 0.288039 |
| zero_missingness_only | zuco1:ZKW | 296 | 15.5405 | 39.8649 | 52.0270 | 0.275912 |
| existing_sentence_ridge | zuco1:ZJS | 216 | 0.4630 | 2.7778 | 4.6296 | 0.025790 |
| existing_sentence_ridge | zuco1:ZKW | 296 | 0.6757 | 2.7027 | 3.7162 | 0.026707 |

## Decision and limitations

Before any fit, useful validation advantage required real MRR to exceed the strongest of the four controls by at least 0.002, positive real-minus-every-control MRR gaps for each reader, and at least two recall cutoffs strictly above all controls. The 0.002 margin is a descriptive practical screen, not a p-value, confidence bound or established effect threshold.

The strongest selected control by aggregate MRR is **count_duration_only**; real-minus-strongest MRR is **-0.065026**.

- `mrr_margin_over_strongest_control`: failed.
- `recall_corroboration`: failed.
- `zuco1:ZJS_mrr_above_all_controls`: failed.
- `zuco1:ZKW_mrr_above_all_controls`: failed.

**No useful validation advantage over the strongest control. Stopped without test extraction or evaluation.**

Two readers and one previously used validation split are not replicated independent cohorts. Words/fixations are not independent sentence-retrieval examples. Supplied spectral preprocessing and gaze-based segmentation remain source assumptions; the new training-only transforms do not undo upstream full-session processing. A gain over random ranking or the old sentence feature recipe can reflect the annotation-derived count/missingness shortcut. It must not be presented as semantic EEG retrieval unless the matched structural controls are beaten consistently. No punctuation repair, extra seed, post-result grid expansion or test evaluation was performed.

## Artifacts and verification

New feature view: `data\processed\zuco1_trt_mean_pilot_v1_20261030`. New fit outputs: `results\spectral_trt_pilot_v1_20261030`. The cache stores fixed-length numeric train/validation vectors, not word text or original EEG arrays. Its index separately records identities/labels for provenance; those fields are never concatenated into model inputs. `gaussian_training_moments.npz`, all mask proofs, source hashes and feature column names are retained. `ridge.npz` files hold selected coefficients and train-fitted imputer/scaler parameters.

Software validation: **7 passed in 5.00 s**. Artifact checks verified six condition reports, all 2,757 Gaussian masks, all candidate/observation lists, selected coefficients and saved ranks. All **547 protected prior result/input files** retained their hashes. The [archive](research_results/spectral_trt_pilot_v1_20261030/README.md) preserves configurations, complete selection grids, coverage/exclusions, validation ranks and participant metrics, source snapshots and source/copy checksums. Large new feature/weight arrays remain in their fresh versioned directories. `validation_checks.json` records separate artifact verification.

Commands executed with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_spectral_pilot.py -q
.\.venv\Scripts\python.exe -u scripts/run_spectral_trt_pilot.py --output results/spectral_trt_pilot_v1_20261030 --view data/processed/zuco1_trt_mean_pilot_v1_20261030
.\.venv\Scripts\python.exe scripts/report_spectral_trt_pilot.py --run results/spectral_trt_pilot_v1_20261030
```

Output paths are guarded against overwrite. Reproduction requires fresh versioned paths.

Final read-only verification reproduced all six saved validation rank arrays
from the stored feature matrices, imputer/scaler parameters and ridge
coefficients. It verified 41 source/copy hashes, original exclusion/split
provenance and the unchanged target-cache model/revision. See the archive's
`final_validation.json` and `provenance_additions.json`. No test-source feature
extraction or test cache array was present.
