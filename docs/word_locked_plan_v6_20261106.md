# Word-locked test: plan v6

**Status: proposed on 2026-11-06, not approved and not run.** [The config](../configs/word_locked_v6_20261106.json) freezes once the owner approves, exact word counts are recorded and the synthetic tests pass. Any later change needs a new version.

This is the last untested route to a real accuracy gain on held-out blocks. [Test v5](research_results/averaged_eeg_v5_20261106/REPORT.md) found no signal in whole-sentence features, but those features ignore timing inside a sentence.

## Question

On held-out list blocks, does EEG locked to each fixated word predict word meaning well enough to tell sentences apart within a block? It has to beat matched noise, shuffled pairs, gaze and word properties, and list position.

## What earlier word work did and did not test

| Earlier work | Setup | What it leaves open |
|---|---|---|
| [Word pretraining v1](research_results/word_pretrain_v1_20261101/) | ZuCo 1.0, eight readers, repeated sentences, word-class labels. Real EEG matched noise, and which words were fixated scored higher than EEG | New sentences on held-out blocks, ZuCo 2.0, and word meaning as a target |
| [Word order pilot](research_results/word_order_pilot_v1_20261031/) and GRU pilots | Word features pooled to sentence targets, on repeated sentences | Same |

So this test does not repeat them. But it starts from a warning: gaze behaviour alone has beaten EEG before, so gaze is a required control here.

## Design

1. **Folds:** the same 13 held-out blocks as v5, each with 10 list positions of buffer on each side. No v1 test sentence or v2 test-zone sentence is stored, fitted or scored.
2. **Input:** the supplied word-level TRT features, which are band amplitudes over each word's fixations. That gives 8 bands by 104 channels (Cz removed), log-transformed and z-scored per reader from training rows only.
3. **Training target (new):** the MiniLM embedding of each word, in a new separate cache. Word text is only ever a target, never an input. Existing targets and caches stay unchanged.
4. **Model:** ridge from one word's EEG to that word's embedding. Alpha is chosen inside each fold's training data.
5. **Sentence prediction:** average each word slot across the readers who fixated it, then average the slots.
6. **Primary score:** expected 10-way within the query's own block, against each candidate's mean word embedding. Chance is 10%.

A sentence's meaning here is roughly the sum of its words, which is crude. The unchanged sentence embedding is scored as a secondary result, so this can be compared with v5.

## Sizes

- **Sentences:** the same as v5, with 419 usable and 255 scored across 13 blocks.
- **Word rows:** about 82,000 usable, with about 51,000 in scored sentences. These are estimates from the 2026-10-30 census, and exact counts will be recorded before freezing.

## Controls

| Arm | What it rules out |
|---|---|
| Matched noise per word | Sentence-level amplitude and slow drift. It keeps each observation's mean EEG and fixation pattern, and removes word-specific EEG |
| Shuffled pairs, mean of 10 | Fitting that does not depend on the sentence-text pairing |
| Gaze and word properties, no EEG | Word length, frequency, reading time, fixations and word position |
| List position, no EEG | Place in the list, even inside a block |
| Partial check | Inside each block, removes linear effects of position, word count, duration and fixated-word count from real and noise predictions |

## Reading

**Clear** means a gap of at least 2 points, with every available 95% lower bound above 0. Lower bounds come from resampling sentences and blocks, and readers too for noise and shuffled pairs. **Beats nulls** means both noise and shuffled pairs are clear.

| Result | Reading | What I would propose next |
|---|---|---|
| Beats nulls, and gaze, position and the partial check are all clear | Word-locked EEG carries sentence information beyond the controls | Confirm on new data under a new plan, then push single-reader and new-reader accuracy |
| Beats nulls, but gaze, position or the partial check is not clear | The gain does not exceed gaze, word properties or position | Report it as a structural result, and move to the audit paper |
| Does not beat nulls | No sentence signal found with word-locked features | Stop chasing decoding on this dataset's local tasks, and put the weight on the audit paper |

Secondary results never change this label:

- **Reader-count curve:** 1, 2, 4, 8 and all readers. One reader is the closest to real single-recording use, and is the accuracy you would actually want to report.
- **Direct word test:** picking the right word out of 10 from a block's own words, for real EEG, noise and gaze.
- **Unchanged sentence targets:** 10-way within block against the existing MiniLM sentence embeddings.
- **All-sentence pool:** 2-, 4-, 10- and 24-way, and each release separately.
- **First-fixation waveform:** mean amplitude in 50 ms bins up to 300 ms, as a rough time-locked input.
- **New readers:** fit on half of each release's readers and score the other half.

## Limits to keep visible

- **Low frequencies are missing from the primary input:** the supplied bands start at 4 Hz, and classic meaning-related brain responses sit largely below that. The waveform input only partly covers this, because a fixation's snippet ends when the eye moves.
- **A fail is not proof:** it would cover these features and a linear model. It would still not prove that the EEG holds no meaning.
- **Gaze-assisted:** words are cut using eye-tracking, so even a pass does not show decoding from EEG alone.
- **Upstream filtering:** the supplied features were filtered over continuous recordings, so nearby trials can mix slightly. This cannot be undone here.
- **Different producers:** matching field names do not prove ZuCo 1.0 and 2.0 were processed identically. Results are reported per release.
- **Skipped words:** about a third of words have no fixation, and which ones is gaze information. The noise and gaze arms keep the same pattern.
- **Seen word types:** many held-out words also occur in training. That is expected for word meaning, and the share is reported.
- **Reused data:** all these sentences have been used before, so this is exploratory and a pass needs confirming. Stream reading also passes over protected word fields, which are discarded without being stored.
- **List order:** sentence ID order is assumed to match presentation order.

## Scope and cost

- **Data:** no download, no CNN training and no final test data. It reads 44 GB of local source files, so the D: drive must stay connected.
- **Checks before running:** synthetic tests must show that a planted word signal is recovered and that noise stays at chance. Position, length, frequency and duration signals must not earn the first reading over 20 seeds each. A run-time leakage audit refuses to continue on any failure.
- **Time:** an estimate of 2 to 4 hours on CPU, not yet measured. Most of it would be reading ZuCo 2.0 source files.
- **Stop rule:** stop after this run whatever the outcome. Scoring the v2 test blocks would need a new plan and the owner's decision.
