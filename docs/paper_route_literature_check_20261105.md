# Literature check for the paper route

Checked on 2026-11-05 with a few web searches and one full-page read. This is a first pass, not a systematic review, and most summaries below come from search results rather than from reading the full papers. Read each paper before citing it.

**Finding: I did not find a paper that directly tests neighbouring-sentence leakage in ZuCo. That does not show none exists.** A systematic search is still needed before calling this idea new.

## What is already published

| Topic | Source | What it means for us |
|---|---|---|
| Noise baselines for ZuCo | [Are EEG-to-Text Models Working?](https://arxiv.org/html/2405.06459v4) | Testing against noise is established, so it is not a new contribution |
| Reader and text leakage across splits | [Rethinking Cross-Subject Data Splitting for Brain-to-Text Decoding](https://arxiv.org/html/2312.10987v2) | Closest related work. It covers reader and sentence overlap, reports generation metrics only and, from reading its page summary, does not address presentation order or time drift |
| Temporal correlation inflating EEG scores | [Li et al. critique, The Perils and Pitfalls of Block Design](https://www.bohrium.com/paper-details/the-perils-and-pitfalls-of-block-design-for-eeg-classification-experiments/812538180229136385-2429), [the rebuttal](https://arxiv.org/pdf/2012.03849) and [Beware of Overestimated Decoding Performance](https://arxiv.org/pdf/2405.17024) | The general problem is known and disputed in size, mainly for visual decoding. Our work would have to be framed as applying it to reading, not discovering it |
| Fixed reading order in ZuCo | [ZuCo 1.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289117/) and [ZuCo 2.0](https://ar5iv.labs.arxiv.org/html/1912.00903) | Same order for all readers, which is what makes the confound possible |
| ZuCo sentence retrieval | [ETER](https://arxiv.org/html/2408.04679v1) and [Brain-CLIPLM](https://arxiv.org/pdf/2604.16370) | Retrieval scores exist. Their metrics and splits differ from ours, so they cannot be compared directly |

## Possible contribution, if v3 supports it

A controlled measurement, on ZuCo sentence retrieval, of how much score comes from training on sentences close in the reading order, holding held-out sentences, input and training size fixed.

v3's near-versus-far design is what would set this apart from a plain split comparison. Its value depends on the v3 result, and it says nothing about new readers.

**Update after v3:** [v3](research_results/proximity_attribution_v3_20261105/REPORT.md) was inconclusive on this measurement. The stronger lead is an exploratory one: the text of neighbouring sentences, with no EEG, scores 53.94% at 10-way on the v1 random split. Searches should now also cover no-EEG neighbouring-text baselines.

## Still to do before writing

1. Run a systematic search, including Google Scholar and citations of the papers above.
2. Read the closest papers in full, especially the data-splitting paper and "Beware of Overestimated Decoding Performance", which I could not read here.
3. Check how the SemKey and GLIM splits treat list position.
4. Review the route with the project supervisor.
