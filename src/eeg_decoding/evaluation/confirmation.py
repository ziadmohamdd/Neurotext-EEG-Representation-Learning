"""Confirmation v10: score the reserved v2 test blocks once with the frozen v5, v6 and v9 pipelines.

Only the layout changes. The four test blocks become the scored folds, every fold trains on the same rows (the whole test
zone and v1 test sentences stay out), and every fold chooses its settings on the same inner blocks, so one fit serves all
four blocks. Buffer sentences around the test blocks enter no fit and no score.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .averaged_eeg import Fold, Layout
from .block_holdout import block_holdout_split
from .word_locked import WordData

CONFIRMED, CONTRADICTED, INCONCLUSIVE = 'Confirmed', 'Contradicted', 'Inconclusive'


def reserved_blocks(records, seed=42, block=25, buffer=10, val_blocks=2, test_blocks=2):
    """Each release's v2 test blocks as sorted lists of sentence ids, and the whole test zone (blocks plus buffers)."""
    splits, _, _ = block_holdout_split(records, seed, block, buffer, val_blocks, test_blocks)
    order = defaultdict(set)
    for r in records:
        order[r.dataset].add(int(r.sentence_id))
    order = {release: sorted(ids) for release, ids in order.items()}
    place = {release: {i: p for p, i in enumerate(ids)} for release, ids in order.items()}
    ids = defaultdict(set)
    for r in splits['test']:
        ids[r.dataset].add(int(r.sentence_id))
    blocks, zone = {}, {}
    for release in sorted(order):
        positions = sorted(place[release][i] for i in ids[release])
        runs, current = [], []
        for p in positions:
            if current and p != current[-1] + 1:
                runs.append(current)
                current = []
            current.append(p)
        runs += [current] if current else []
        if any(len(run) != block for run in runs) or len(runs) != test_blocks:
            raise ValueError(f'Test blocks in {release} are not {test_blocks} runs of {block}')
        blocks[release] = [[order[release][p] for p in run] for run in runs]
        near = {p + d for run in runs for p in run for d in range(-buffer, buffer + 1)}
        zone[release] = {order[release][p] for p in near if 0 <= p < len(order[release])}
    return blocks, zone


class ConfirmLayout(Layout):
    """Rows outside the buffers and v1 test; folds are the test blocks; training excludes every test block."""
    def __init__(self, records, blocks, zone, v1_test, forbidden, block=25, buffer=10, minimum=10):
        self._blocks = blocks
        tested = {(release, i) for release, runs in blocks.items() for run in runs for i in run}
        buffers = {release: {i for i in ids if (release, i) not in tested} for release, ids in zone.items()}
        super().__init__(records, buffers, v1_test, forbidden, block, buffer, minimum)
        self.zone = zone
        self.test_sentences = np.array(sorted({self.lookup[k] for k in tested if k in self.lookup}))
        self.test_row = np.isin(self.sentence, self.test_sentences)

    def _tile(self):
        folds = []
        for release in self.releases:
            for run in self._blocks.get(release, []):
                positions = [self.place[release][i] for i in run]
                members = [self.lookup[release, i] for i in run if (release, i) in self.lookup and i not in self.forbidden.get(release, ())]
                if len(members) < self.minimum:
                    raise ValueError(f'A test block in {release} keeps only {len(members)} scoreable sentences')
                folds.append(Fold(release, min(positions), max(positions), np.array(members)))
        return folds

    def train_mask(self, index):
        return ~self.test_row

    def inner_blocks(self, index, seed):
        """The same inner blocks for every fold, drawn as for fold 0, never inside the test zone."""
        return super().inner_blocks(0, seed)


@dataclass
class ConfirmWordData(WordData):
    """Word rows over a ConfirmLayout: every fold trains on sentences outside all test blocks."""
    def train_sentences(self, index):
        return ~np.isin(np.arange(len(self.layout.identities)), self.layout.test_sentences)


def confirm_audit(layout, seed):
    """Checks specific to this layout, on top of the shared fold audit."""
    failures = []
    for index, fold in enumerate(layout.folds):
        train = layout.train_mask(index)
        if (train & layout.test_row).any():
            failures.append(f'Fold {index}: a test-block row is in training')
        for block in layout.inner_blocks(index, seed):
            if np.isin(block.sentences, layout.test_sentences).any():
                failures.append(f'Fold {index}: an inner block holds a test sentence')
            for release, ids in layout.zone.items():
                if block.release == release and any(layout.order[release][p] in ids for p in range(block.low, block.high + 1)):
                    failures.append(f'Fold {index}: an inner block touches the test zone')
    in_zone = {(r.dataset, int(r.sentence_id)) for r in layout.rows if int(r.sentence_id) in layout.zone.get(r.dataset, ())}
    tested = {layout.identities[s] for s in layout.test_sentences}
    if in_zone - tested:
        failures.append('A buffer sentence is in the usable rows')
    scored = {layout.identities[s] for s in layout.scored}
    if scored != tested:
        failures.append('Scored sentences differ from the test-block sentences')
    return failures


def clear(gap, bounds, minimum=.02):
    """At least `minimum` and every available lower bound above 0."""
    return gap >= minimum and all(low > 0 for low, _ in bounds)


def read_claims(c1, c2, c3, minimum=.02, margin=.03):
    """Label the three claims. Each input holds a gap and its available (low, high) intervals, as fractions.

    c1: real minus noise per pipeline. c2: real word-locked EEG minus gaze. c3: gaze plus real minus gaze plus noise.
    """
    labels = {}
    if any(clear(g['gap'], g['bounds'], minimum) for g in c1.values()):
        labels['C1'] = CONTRADICTED
    else:
        labels['C1'] = CONFIRMED
    gaze_over_real = [(-high, -low) for low, high in c2['bounds']]
    if clear(-c2['gap'], gaze_over_real, minimum):
        labels['C2'] = CONFIRMED
    elif clear(c2['gap'], c2['bounds'], minimum):
        labels['C2'] = CONTRADICTED
    else:
        labels['C2'] = INCONCLUSIVE
    if clear(c3['gap'], c3['bounds'], minimum):
        labels['C3'] = CONTRADICTED
    elif all(high < margin for _, high in c3['bounds']):
        labels['C3'] = CONFIRMED
    else:
        labels['C3'] = INCONCLUSIVE
    return labels


__all__ = ['CONFIRMED', 'CONTRADICTED', 'INCONCLUSIVE', 'ConfirmLayout', 'ConfirmWordData', 'clear', 'confirm_audit', 'read_claims', 'reserved_blocks']
