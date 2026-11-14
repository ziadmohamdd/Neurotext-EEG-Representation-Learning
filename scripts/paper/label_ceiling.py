"""Expected N-way accuracy of a predictor that knows only a sentence's group (corpus or task) and is at chance within it.

Distractors are N-1 sentences drawn uniformly without replacement from the other S-1 sentences of the pool. The predictor
ranks every same-group sentence above every other-group one and breaks ties within the group at random, so the score is
E[1 / (1 + M)], with M the hypergeometric count of same-group distractors. No data is read; group sizes come from the
GLIM output check (156 sentences after protected sentences were removed)."""
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = {'ZuCo 1.0 movie reviews': 40, 'ZuCo 1.0 normal reading': 15, 'ZuCo 1.0 task-specific': 39,
         'ZuCo 2.0 normal reading': 24, 'ZuCo 2.0 task-specific': 38}
CORPORA = {'movie reviews': 40, 'biographies': 116}
GLIM = {2: 67.26, 4: 42.71, 10: 21.46, 24: 10.47}
GLIM_NOISE = {2: 51.55, 4: 26.50, 10: 11.26, 24: 5.06}


def group_score(groups, n):
    total = sum(groups.values())
    score = 0.0
    for k in groups.values():
        same, other = k - 1, total - k
        p = sum(comb(same, m) * comb(other, n - 1 - m) / comb(total - 1, n - 1) / (1 + m) for m in range(0, min(same, n - 1) + 1))
        score += k * p
    return 100 * score / total


def main():
    rows = [dict(n=n, chance=round(100 / n, 2), corpus_only=round(group_score(CORPORA, n), 2), task_only=round(group_score(TASKS, n), 2),
                 glim=GLIM[n], glim_noise=GLIM_NOISE[n]) for n in (2, 4, 10, 24)]
    out = ROOT/'docs/research_results/paper_analyses_20261114'
    out.mkdir(exist_ok=True)
    (out/'label_ceiling.json').write_text(json.dumps(dict(groups=dict(tasks=TASKS, corpora=CORPORA), rows=rows), indent=2), encoding='utf-8')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
