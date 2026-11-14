"""Draw the paper figures from saved result JSONs only. No EEG is read and nothing is fitted."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'docs/research_results'
OUT = ROOT/'docs/paper_figures'
REAL, NOISE, SHUF, NOEEG, CEIL, INK = '#1f4e79', '#8c8c8c', '#c8c8c8', '#d9822b', '#2a9d8f', '#333333'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})


def load(path):
    return json.loads((R/path).read_text(encoding='utf-8'))


def chance(ax, y, label='Chance'):
    ax.axhline(y, color=INK, lw=0.8, ls=':', zorder=0)
    ax.text(ax.get_xlim()[1], y, f' {label}', va='center', fontsize=7.5, color=INK)


def bars(ax, labels, values, colors):
    b = ax.bar(range(len(values)), values, color=colors, width=0.7)
    for rect, v in zip(b, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f'{v:.1f}', ha='center', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(values)), labels)


def fig1():
    c = load('paper_analyses_20261114/label_ceiling.json')['rows']
    pairs = load('glim_output_check_20261106/task_pair_scores.json')
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw=dict(width_ratios=[1, 1.4]))
    n = [r['n'] for r in c]
    for key, col, lab, mk in [('glim', REAL, 'GLIM outputs', 'o'), ('corpus_only', CEIL, 'Knows corpus only, no EEG', 's'),
                              ('glim_noise', NOISE, 'GLIM, noise input', 'o'), ('chance', INK, 'Chance', '')]:
        a.plot(n, [r[key] for r in c], color=col, marker=mk or None, ms=4, lw=1.4, ls=':' if key == 'chance' else '-', label=lab)
    a.set_xscale('log'); a.set_xticks(n, [str(x) for x in n]); a.minorticks_off()
    a.set_xlabel('N-way'); a.set_ylabel('Expected accuracy (%)'); a.set_title('(a) Mixed-task retrieval', loc='left', fontsize=9)
    a.legend(frameon=False, fontsize=7)
    keys = list(pairs['glim'])
    short = {'ZuCo1_task1': 'Rev', 'ZuCo1_task2': '1-NR', 'ZuCo1_task3': '1-TSR', 'ZuCo2_task2': '2-NR', 'ZuCo2_task3': '2-TSR'}
    keys.sort(key=lambda k: (not k.startswith('ZuCo1_task1'), k))
    x = range(len(keys))
    b.bar([i - 0.2 for i in x], [100 * pairs['glim'][k] for k in keys], 0.4, color=REAL, label='GLIM outputs')
    b.bar([i + 0.2 for i in x], [100 * pairs['noise_input'][k] for k in keys], 0.4, color=NOISE, label='Noise input')
    b.set_xticks(list(x), [' vs '.join(short[t] for t in k.split('|')) for k in keys], fontsize=6.5, rotation=45, ha='right', rotation_mode='anchor')
    b.axvline(3.5, color=INK, lw=0.6)
    b.text(1.5, 100, 'Reviews vs\nbiographies', ha='center', va='bottom', fontsize=7); b.text(6.5, 100, 'Biography tasks\nvs each other', ha='center', va='bottom', fontsize=7)
    b.set_ylim(0, 115); b.set_ylabel('2-way accuracy (%)'); b.set_title('(b) 2-way by pair of tasks', loc='left', fontsize=9, pad=16)
    chance(b, 50); b.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0.45, 1.0), ncol=2)
    fig.savefig(OUT/'fig1_task_mixing.png'); plt.close(fig)


def fig2():
    s = load('position_bridge_v4_20261105/scores.json')
    rows = [('Neighbouring\ntext, no EEG', 'neighbour_text', NOEEG), ('CNN,\nrandom split', 'v1_cnn', REAL),
            ('Position from\nreal EEG', 'bridge_v1_real', REAL), ('Position from\nmatched noise', 'bridge_v1_gaussian', NOISE),
            ('Position,\nshuffled', 'bridge_v1_shuffled', SHUF)]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    bars(ax, [r[0] for r in rows], [100 * s[r[1]]['10_way'] for r in rows], [r[2] for r in rows])
    ax.tick_params(axis='x', labelsize=7.5); ax.set_ylabel('10-way accuracy (%)'); chance(ax, 10)
    ax.set_title('Random split, 53 validation sentences, familiar readers', loc='left', fontsize=9)
    fig.savefig(OUT/'fig2_position.png'); plt.close(fig)


def fig3():
    tests = [('Whole-sentence\n(v5)', 'averaged_eeg_v5_20261106', 'timing', 'Timing'),
             ('Word-locked\n(v6)', 'word_locked_v6_20261106', 'gaze_word', 'Gaze'),
             ('First fixation\n(v9)', 'first_fixation_v9_20261114', 'gaze_word', 'Gaze')]
    arms = [('real', 'Real EEG', REAL), ('gaussian', 'Matched noise', NOISE), ('class_shuffled', 'Shuffled pairs', SHUF), (None, 'Best no-EEG control', NOEEG)]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    w = 0.2
    for j, (key, lab, col) in enumerate(arms):
        vals = [100 * load(f'{f}/decision.json')['primary'][key or ctrl] for _, f, ctrl, _ in tests]
        xs = [i + (j - 1.5) * w for i in range(len(tests))]
        ax.bar(xs, vals, w, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.4, f'{v:.1f}', ha='center', fontsize=6.5, color=INK)
    ax.set_xticks(range(len(tests)), [t[0] for t in tests]); ax.set_ylabel('10-way within block (%)')
    ax.set_title('Held-out blocks, 255 sentences, all readers averaged', loc='left', fontsize=9)
    chance(ax, 10); ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
    ax.set_ylim(0, 32)
    fig.savefig(OUT/'fig3_block_tests.png'); plt.close(fig)


def fig4():
    tests = [('Whole-sentence (v5)', 'averaged_eeg_v5_20261106'), ('Word-locked (v6)', 'word_locked_v6_20261106'), ('First fixation (v9)', 'first_fixation_v9_20261114')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    ks = ['1', '2', '4', '8', 'all']
    for ax, (title, f) in zip(axes, tests):
        rc = load(f'{f}/secondary.json')['reader_curve']
        ax.plot(range(5), [100 * rc['real'][k] for k in ks], color=REAL, marker='o', ms=3.5, label='Real EEG')
        ax.plot(range(5), [100 * rc['gaussian'][k] for k in ks], color=NOISE, marker='o', ms=3.5, label='Matched noise')
        ax.set_xticks(range(5), ['1', '2', '4', '8', 'All']); ax.set_title(title, loc='left', fontsize=9)
        ax.axhline(10, color=INK, lw=0.8, ls=':')
    axes[0].set_ylabel('10-way within block (%)'); axes[1].set_xlabel('Readers averaged'); axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(OUT/'fig4_readers.png'); plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
    print(sorted(p.name for p in OUT.iterdir()))
