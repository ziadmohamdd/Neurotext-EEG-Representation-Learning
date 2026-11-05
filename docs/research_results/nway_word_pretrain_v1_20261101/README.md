# Expected N-way retrieval accuracy — latest word-pretrained GRU

These are exact expectations under uniformly sampled distractors, computed from replayed frozen models on the original eight training-reader CV folds. They are not new independent validation/test results, Monte Carlo estimates or improvements to the trained model.

Each pool contains the correct sentence and N−1 unique distractors drawn without replacement from the original 299 incorrect sentences. For b strictly lower-scoring candidates out of M=300, P(strict top-1)=C(b,N−1)/C(M−1,N−1). Ties with the truth count as a miss, matching the original average-tied-rank<=1 criterion. All arms use the same candidate universe and uniform sampling mechanism; exact expectation removes sampling variability. The probabilities are averaged within reader then equally across the eight readers (2,245 total observations).

Random chance is 1/N. Uniform random distractors do not control sentence length or semantic similarity. Hard-negative or length-matched pools would be a different evaluation. Repeated candidate subsets are not new independent participants or cohorts. No setting was selected from these N-way scores, and no raw source, outer-validation or test EEG array was used.

## Expected top-1 accuracy (%)

| N | Pretrained real EEG | Gaussian | Class-shuffled | Missingness-only | Structural-only | Chance |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 49.2079 | 49.2034 | 50.2292 | 53.8270 | 90.4389 | 50.0000 |
| 5 | 19.5309 | 18.8058 | 20.2595 | 23.7697 | 75.0557 | 20.0000 |
| 10 | 9.7124 | 8.8002 | 10.2744 | 12.8284 | 62.8101 | 10.0000 |
| 20 | 4.7439 | 4.0531 | 5.2801 | 7.0310 | 51.3648 | 5.0000 |
| 50 | 1.8078 | 1.4160 | 2.2851 | 3.2405 | 37.8816 | 2.0000 |
| 100 | 0.8882 | 0.6483 | 1.2739 | 1.8675 | 28.5345 | 1.0000 |
| 300 | 0.3946 | 0.2160 | 0.5783 | 0.8977 | 16.1129 | 0.3333 |

Fresh equal-update and same-sentence-update baseline results, all reader-level values and exact score comparison counts are also archived. Smaller N raises chance and changes difficulty; higher raw percentages must not be presented as improved EEG decoding.

Three focused analytical tests passed before execution. Replays reproduced all 15,715 original ranks and every original 300-way macro Recall@1. All 2684 protected prior files remained SHA256-identical. No fitting, tuning or new checkpoint selection occurred. The .venv and prior reports/results are unchanged.
