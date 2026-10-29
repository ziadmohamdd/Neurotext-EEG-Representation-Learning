# Unseen-subject and unseen-sentence ridge results

User-reported Windows runs, consolidated 2026-10-29. Seeds 42, 43 and 44 use
the duration-filtered 105-channel manifests (3,349 ZuCo1 and 5,884 ZuCo2 source
observations), full recordings, and train-only normalization and ridge scaling.
Subjects and canonical sentence texts are disjoint across training, validation
and testing. Gaussian and shuffled models are separately trained with matching
split assignments, candidate pools and validation selection rules per seed.

| Dataset | Training condition | Mean Recall@1 (%) | Mean MRR | Sample SD of MRR |
|---|---|---:|---:|---:|
| ZuCo1 | Real EEG | 2.335 | 0.085268 | 0.007368 |
| ZuCo1 | Gaussian inputs | 2.335 | 0.086335 | 0.009157 |
| ZuCo1 | Shuffled training pairs | 2.895 | 0.090252 | 0.014689 |
| ZuCo2 | Real EEG | 1.337 | 0.073448 | 0.006412 |
| ZuCo2 | Gaussian inputs | 1.691 | 0.075441 | 0.003512 |
| ZuCo2 | Shuffled training pairs | 1.348 | 0.068180 | 0.011447 |

These are unweighted arithmetic means over three splits. SD is sample SD
(`ddof=1`), not standard error or a confidence interval. Repeated splits share
participants and stimuli and are not independent samples of a new population.
No significance claim follows from these three means or SD values.

ZuCo1 has 60 candidates in each split; uniform random-ranking MRR is 0.077998
and Recall@1 is 1.667%. Test query counts are 117, 111 and 115 for seeds 42–44.
ZuCo2 has 68 candidates per split; chance MRR is 0.070648 and Recall@1 is
1.471%. Query counts are 197, 203 and 193. Candidate identities can differ
between seeds. Neither raw percentages nor MRR should be compared directly
against the earlier 300/344-candidate subject-only evaluations.

Real EEG exceeds Gaussian MRR only on seed 43 in each dataset. It does not
show a consistent advantage across the three reported splits. Gaussian and
shuffled controls sometimes exceed analytic random-ranking chance, so a score
above that expectation alone does not demonstrate useful EEG information.
These results do not establish reliable generalization to unseen sentence
content, and do not demonstrate that no EEG semantic information exists.

Provenance: seed-42 metrics were provided in prior terminal messages; all 12
seed-43/44 metrics were supplied together in an uploaded terminal transcript.
Output suffix for seed-42 real runs is `20261029_025322`, seed-42 controls
`20261029_032352`, and seed-43/44 runs `20261029_034919`. The user's latest
export contains 63 executed evaluation reports, including controls, corruptions,
and repeat/superseded reports; it is not a count of 63 independent methods.
The complete CSV and underlying result JSONs should be archived with the final
report for traceability. The cloud did not rerun these real-data experiments.
