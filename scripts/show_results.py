"""Print every main result of the project from saved files only. Fits, evaluates and downloads nothing.

Usage:  .venv\\Scripts\\python.exe scripts\\show_results.py            (print all results)
        .venv\\Scripts\\python.exe scripts\\show_results.py --figures  (also open the five paper figures)
Scores are expected N-way accuracy in percent; chance is 1/N (10% at 10-way).
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'docs/research_results'


def load(path):
    return json.loads((R / path).read_text(encoding='utf-8'))


def pct(x):
    return f'{100 * x:6.2f}'


def heading(text):
    print(f'\n{"=" * 78}\n{text}\n{"=" * 78}')


def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: '  '.join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    print(line(header)); print('  '.join('-' * w for w in widths))
    for r in rows:
        print(line(r))


def interval(lo_hi):
    return f'{100 * lo_hi[0]:.2f} to {100 * lo_hi[1]:.2f}'


def finding1():
    heading('FINDING 1  Task mixing: GLIM public outputs vs a corpus-only predictor (no EEG)')
    rows = load('paper_analyses_20261114/label_ceiling.json')['rows']
    keys = [('chance', 'Chance'), ('glim', 'GLIM outputs, our scoring'), ('glim_noise', 'GLIM, noise input'),
            ('corpus_only', 'Knows corpus only, no EEG'), ('task_only', 'Knows task only, no EEG')]
    table(['Predictor'] + [f'{r["n"]}-way' for r in rows], [[label] + [f'{r[k]:.2f}' for r in rows] for k, label in keys])
    v8 = load('public_outputs_check_v8_20261112/decision.json')
    print('\nTwo more published models (10-way, mixed pool):')
    for name, label in (('bart', 'Wang & Ji BART'), ('t5', 'Jo et al. T5')):
        s = v8[name]['scores']
        print(f'  {label:16} real {s["real"]["mixed"]:.2f}   noise input {s["noise"]["mixed"]:.2f}   corpus only {s["corpus_label_only"]["mixed"]:.2f}   -> {v8[name]["label"]}')


def finding2():
    heading('FINDING 2  Random split leaks list position (v4, 53 validation sentences, familiar readers)')
    s = load('position_bridge_v4_20261105/scores.json')
    rows = [('Neighbouring text, no EEG', 'neighbour_text'), ('CNN, random split', 'v1_cnn'), ('Position from real EEG', 'bridge_v1_real'),
            ('Position from matched noise', 'bridge_v1_gaussian'), ('Position, shuffled', 'bridge_v1_shuffled')]
    table(['Predictor', '2-way', '10-way'], [[label, pct(s[k]['2_way']), pct(s[k]['10_way'])] for label, k in rows])
    print('  (The paper also averages the neighbouring-text baseline over 20 random splits: see position_bridge_v4_20261105/REPORT.md.)')


def finding3():
    heading('FINDING 3  Held-out list blocks, 10-way within block, all readers averaged')
    tests = [('v5 whole-sentence', 'averaged_eeg_v5_20261106', 'timing'), ('v6 word-locked', 'word_locked_v6_20261106', 'gaze_word'),
             ('v9 first fixation', 'first_fixation_v9_20261114', 'gaze_word'), ('v7 same sentence', 'same_sentence_v7_20261106', 'position')]
    rows = []
    for label, folder, ctrl in tests:
        d = load(f'{folder}/decision.json')
        p = d['primary']
        rows.append([label, pct(p['real']), pct(p['gaussian']), pct(p['class_shuffled']), f'{ctrl} {pct(p[ctrl]).strip()}',
                     interval(d['intervals']['gaussian']['sentence_interval']), d['label']])
    table(['Test', 'Real', 'Noise', 'Shuffled', 'Strongest no-EEG', 'Real-noise 95% (sent.)', 'Verdict'], rows)
    v9 = load('first_fixation_v9_20261114/decision.json')['primary']
    print(f'\n  v9 gaze + real EEG {pct(v9["gaze_plus_real"]).strip()}  vs  gaze + noise {pct(v9["gaze_plus_noise"]).strip()}')


def limits():
    heading('EFFECT LIMITS  Largest gain the 95% intervals still allow (points, 10-way)')
    table(['Test', 'Comparison', 'Gap', 'Largest gain allowed'],
          [[r['test'], r['comparison'], f'{r["gap_points"]:.2f}', f'{r["largest_upper"]:.2f}'] for r in load('paper_analyses_20261114/effect_limits.json')])


def confirmation():
    heading('CONFIRMATION v10  Scored once on 85 never-used sentences (frozen plan)')
    c = load('confirmation_v10_20261114/claims.json')
    rows = []
    for v in ('v5', 'v6', 'v9'):
        dev, test = c['development_primary'][v], c['test_primary'][v]
        rows.append([v, pct(dev['real']), pct(dev['gaussian']), pct(test['real']), pct(test['gaussian'])])
    table(['Pipeline', 'Dev real', 'Dev noise', 'Test real', 'Test noise'], rows)
    t6, t9 = c['test_primary']['v6'], c['test_primary']['v9']
    print(f'\n  Gaze, no EEG {pct(t6["gaze_word"]).strip()}   gaze + EEG {pct(t9["gaze_plus_real"]).strip()}   gaze + noise {pct(t9["gaze_plus_noise"]).strip()}')
    names = {'C1': 'EEG does not beat matched noise', 'C2': 'Gaze beats word-locked EEG', 'C3': 'EEG adds little to gaze'}
    print()
    for k, label in c['labels'].items():
        print(f'  {k}  {names[k]:34} -> {label}')


def history():
    heading('EARLIER WORK and PLANS')
    print('  Earlier decoders (CNN, SSL, LaBraM, GRU; scored as MRR, not accuracy): docs/research_results/REPORT.md')
    plan = ROOT / 'docs/replication_plan_v11_20261121.md'
    counts = R.parent / 'replication_v11/sentence_counts.json'
    if plan.exists() and counts.exists():
        n = json.loads(counts.read_text(encoding='utf-8'))
        kept = {k: v['kept_if_repeats_also_dropped'] for k, v in n.items() if isinstance(v, dict) and 'kept_if_repeats_also_dropped' in v}
        print(f'  v11 replication: PLAN ONLY, not run, no EEG downloaded. Fresh sentences available: {kept}')
    print('  Paper: docs/paper_draft_20261114.md   Deck: docs/presentation/NeuroText_final_v3_20261117.pptx')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figures', action='store_true', help='also open the five paper figures')
    args = parser.parse_args()
    print('NeuroText results (saved files only; expected N-way accuracy in %, chance = 100/N)')
    for part in (finding1, finding2, finding3, limits, confirmation, history):
        part()
    if args.figures:
        for fig in sorted((ROOT / 'docs/paper_figures').glob('fig*.png')):
            print('opening', fig.name)
            os.startfile(fig)  # Windows: opens with the default image viewer


if __name__ == '__main__':
    main()
