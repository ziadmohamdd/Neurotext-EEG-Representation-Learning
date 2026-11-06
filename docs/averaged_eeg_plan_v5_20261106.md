# Averaged-EEG test: plan v5

**Status: proposed on 2026-11-06, not approved and not run.** [The config](../configs/averaged_eeg_v5_20261106.json) freezes once the owner approves and the synthetic tests pass. Any later change needs a new version.

This plan does not try to build a usable decoder. It asks whether there is sentence information worth chasing on the fair test, where [v2](research_results/block_holdout_pilot_v2_20261105/REPORT.md) stayed at chance.

## Question

On held-out list blocks, does EEG averaged across readers beat matched noise, shuffled pairs, timing and list position at telling sentences apart within a block?

**Why averaging:** the same sentence was read by 8 to 18 readers. Averaging keeps what readers share and cancels what they do not, so it gives single-reader models their best chance.

**Not done before:** a search of the repo's docs, configs and code found no averaging of EEG across readers.

## Design

1. **Folds:** each of 13 list blocks is held out once (5 in ZuCo 1.0, 8 in ZuCo 2.0). Training leaves out the block and 10 list positions on each side, and uses all other usable sentences from both releases.
2. **Input:** the existing statistical EEG features, with a 50 Hz notch, Cz removed and the whole sentence (cropped only above 30 seconds). Features are z-scored per reader and fold, from that reader's training rows only.
3. **Model:** ridge from single-reader training rows to the unchanged MiniLM text embeddings. Alpha is chosen inside each fold's training data, from a held-back block with its own buffer.
4. **Scoring:** for each held-out sentence, average all available readers' features and predict once. For a linear model this equals averaging the readers' predictions.
5. **Primary score:** expected 10-way within the query's own block, one equal-weight score per sentence. Chance is 10%. Staying inside a block makes coarse list position much less useful than it is across the list.

## Sizes

These come from the manifests only. No EEG was read to get them.

| | ZuCo 1.0 | ZuCo 2.0 | Total |
|---|---:|---:|---:|
| Usable sentences | 186 | 233 | 419 |
| Scored sentences | 105 | 150 | 255 |
| Scored observations | 1,206 | 2,509 | 3,715 |
| Scored blocks | 5 | 8 | 13 |
| Readers | 12 | 18 | 30 |
| Readers per scored sentence (lowest) | 8 | 14 | |

All 6,028 usable observations leave out v1 test sentences and the v2 test zone. Using 13 blocks scores 255 sentences, against 100 in v2's validation.

## Controls

| Arm | What it rules out |
|---|---|
| Matched noise, averaged the same way | Channel amplitudes alone, including slow changes over a session |
| Shuffled pairs, mean of 10 | Fitting that does not depend on the sentence-text pairing |
| Timing and word count, no EEG | Sentence length and reading time |
| List position, no EEG | Place in the list, even inside a block |

## Reading

**Beats nulls** means gaps of at least 2 points over noise and over shuffled pairs. Both gaps also need 95% lower bounds above 0 when resampling sentences, blocks and readers.

| Result | Reading | What I would propose next |
|---|---|---|
| Beats nulls, and at least 2 points above timing and position | Averaged EEG carries sentence information beyond the controls | Try to carry it to single readers (ideas 2 and 3), then new readers |
| Beats nulls, but within 2 points of timing or position | The gain does not exceed timing or position | Remove that route, or stop |
| Does not beat nulls | No sentence signal found with these features | Word-locked decoding only, and put the weight on the audit paper |

Secondary results never change this label:

- **Reader-count curve:** score at 1, 2, 4, 8 and all readers, for real EEG and noise. A real shared signal should rise with readers and rise faster than noise.
- **All-sentence pool:** 2-, 4-, 10- and 24-way against all scored sentences in the release.
- **Each release:** ZuCo 1.0 and ZuCo 2.0 separately.
- **Time-binned input:** mean amplitude in 20 equal parts of each sentence, which keeps a rough time course.
- **Fit on sentence means:** in place of single-reader rows.
- **New readers:** fit on half of each release's readers and average the other half, on the same held-out blocks. This is the first test here of new readers and new sentences together.

## Limits to keep visible

- **Not a decoder:** real use means one person's single recording, so even a clear pass does not raise a usable accuracy.
- **A fail is narrow:** these features summarise a whole sentence, and readers reach each word at different times. A fail does not rule out a word-locked signal, or prove that the EEG holds no meaning.
- **Drift gets averaged too:** readers share a list order, so averaging can sharpen a position signal as well. This is why the primary score stays within a block and list position is a control.
- **Reader mix:** some readers miss whole runs, so the readers averaged differ between blocks. Scoring within a block and per-reader scaling reduce this, but do not remove it.
- **Reused data:** all these sentences have been used before, including v2's validation blocks, so this is exploratory and a pass needs confirming.
- **List order:** sentence ID order is assumed to match presentation order, and has not been checked against stimulus logs.
- **Power:** with 255 sentences and 13 blocks, a small effect could be missed. The block interval is the widest and may decide the result.

## Scope and cost

- **Data:** no download, no retraining of the CNN and no final test data. v1 test sentences, v2 test blocks and their buffers are never read.
- **Checks before running:** synthetic tests must show that a planted signal is recovered and that noise stays at chance. They must also show that position-only and duration-only signals cannot earn the first reading. A run-time leakage audit refuses to continue on any failure.
- **Time:** an estimate of 1 to 2 hours on CPU, not yet measured.
- **Stop rule:** stop after this run whatever the outcome. Scoring the v2 test blocks would need a new plan and the owner's decision.
