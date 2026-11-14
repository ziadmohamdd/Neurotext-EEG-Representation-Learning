"""Largest EEG gain each saved test still allows: upper ends of the saved 95% intervals. Reads decision.json files only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'docs/research_results'
TESTS = {'v5 whole-sentence': 'averaged_eeg_v5_20261106', 'v6 word-locked': 'word_locked_v6_20261106',
         'v7 same sentence': 'same_sentence_v7_20261106', 'v9 first fixation': 'first_fixation_v9_20261114'}
COMPARISONS = {'gaussian': 'real minus matched noise', 'partial': 'partial check, real minus noise', 'adds_to_gaze': 'gaze plus real minus gaze plus noise'}


def main():
    rows = []
    for test, folder in TESTS.items():
        d = json.loads((R/folder/'decision.json').read_text(encoding='utf-8'))
        for key, name in COMPARISONS.items():
            if key not in d['intervals']:
                continue
            iv = d['intervals'][key]
            bounds = {k.replace('_interval', ''): [round(100*v, 2) for v in iv[k]] for k in iv if k.endswith('_interval')}
            rows.append(dict(test=test, comparison=name, gap_points=round(100*iv['estimate'], 2), intervals=bounds,
                             largest_upper=max(b[1] for b in bounds.values()), smallest_upper=min(b[1] for b in bounds.values())))
    out = R/'paper_analyses_20261114'
    out.mkdir(exist_ok=True)
    (out/'effect_limits.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    for r in rows:
        print(r['test'], '|', r['comparison'], '|', r['gap_points'], '|', r['intervals'], '| max upper', r['largest_upper'])


if __name__ == '__main__':
    main()
