"""Ask whether the first-fixation waveform tells held-out sentences apart beyond noise, gaze and the channels nearest the eyes."""
import numpy as np
from .averaged_eeg import resample_gap
from .word_locked import (Table, WordScored, gaze_options, noise_options, resample_readers, run_arm, shuffled_table, within_score)

PRIMARY = 10
FIRST = 'The first-fixation waveform carries sentence information beyond noise, gaze and eye channels'
SECOND = 'The gain does not exceed gaze, eye channels or structure'
THIRD = 'The first-fixation result did not hold'
BINS = 6


def without_channels(waves, channels, removed):
    """Drop every bin of the removed channels. Columns run bin by bin, with every channel inside each bin."""
    keep = ~np.isin(np.arange(waves.shape[1]) % channels, removed)
    return waves[:, keep]


def with_gaze(gaze, eeg):
    """One feature matrix per fold: gaze columns beside the EEG option, both scaled per reader inside the fold."""
    return lambda index: [np.c_[gaze(index)[0], eeg(index)[0]]]


def read_outcome(gaps, lower_bounds, minimum=.02):
    """Frozen reading. A gap is clear at the minimum with every available lower bound above 0."""
    clear = lambda names: all(gaps[n] >= minimum and all(v > 0 for v in lower_bounds[n].values()) for n in names)
    beats = clear(('gaussian', 'class_shuffled'))
    label = THIRD if not beats else FIRST if clear(('adds_to_gaze', 'not_eye_channels', 'partial')) else SECOND
    return {'label': label, 'beats_nulls': beats, 'gaps': gaps, 'lower_bounds': lower_bounds, 'minimum_gap': minimum}


def run_test(data, waves, channels, anterior, plan, status=print, iterations=10000, reader_iterations=2000):
    """Every arm and reading in the frozen plan. `anterior` holds column indices of the channels removed in the eye-channel check."""
    seed = plan['seed']
    alphas, shuffles = plan['model']['alphas'], plan['arms_shuffles']
    base = Table(data)
    true = lambda index: [base]
    with_shuffles = lambda index: [base] + [shuffled_table(data, index, seed + s) for s in range(shuffles)]
    features = lambda matrix: (lambda index: [matrix])
    status('FIT real waveform and shuffled pairs')
    rows, predicted, chosen = run_arm(data, features(waves), with_shuffles, alphas, seed)
    real, shuffled = predicted[0], predicted[1:]
    scored = WordScored(data, rows)
    selection = {'real': [c[0] for c in chosen]}
    noise = noise_options(data, waves, seed)
    gaze = gaze_options(data)
    front = without_channels(waves, channels, anterior)
    arms = {'real': real}
    for name, options, scale in (('gaussian', noise, True), ('gaze_word', gaze, False), ('gaze_plus_real', with_gaze(gaze, features(waves)), True),
                                 ('gaze_plus_noise', with_gaze(gaze, noise), True), ('anterior_removed', features(front), True),
                                 ('anterior_removed_noise', noise_options(data, front, seed), True)):
        status(f'FIT {name}')
        _, predicted, chosen = run_arm(data, options, true, alphas, seed, scale=scale)
        arms[name] = predicted[0]
        selection[name] = [c[0] for c in chosen]
    values = {name: scored.within(scored.average(p)) for name, p in arms.items()}
    values['class_shuffled'] = np.mean([scored.within(scored.average(p)) for p in shuffled], axis=0)
    covariates = scored.covariates()
    for name in ('real', 'gaussian'):
        values[f'{name}_partial'] = scored.within(scored.partial(scored.average(arms[name]), covariates))
    primary = {name: float(v.mean()) for name, v in values.items()}
    status(f'Primary 10-way within block: {primary}')
    pairs = {'gaussian': ('real', 'gaussian'), 'class_shuffled': ('real', 'class_shuffled'), 'adds_to_gaze': ('gaze_plus_real', 'gaze_plus_noise'),
             'not_eye_channels': ('anterior_removed', 'anterior_removed_noise'), 'partial': ('real_partial', 'gaussian_partial')}
    gaps = {name: primary[a] - primary[b] for name, (a, b) in pairs.items()}
    intervals = {name: resample_gap(values[a] - values[b], scored.block_of, iterations, seed) for name, (a, b) in pairs.items()}
    status('RESAMPLE readers')
    readers = resample_readers(scored, real, {'gaussian': [arms['gaussian']], 'class_shuffled': shuffled}, reader_iterations, seed)
    readers.update(resample_readers(scored, arms['gaze_plus_real'], {'adds_to_gaze': [arms['gaze_plus_noise']]}, reader_iterations, seed))
    readers.update(resample_readers(scored, arms['anterior_removed'], {'not_eye_channels': [arms['anterior_removed_noise']]}, reader_iterations, seed))
    lower = {name: {'sentence': intervals[name]['sentence_interval'][0], 'block': intervals[name]['block_interval'][0]} for name in gaps}
    for name in readers:
        lower[name]['reader'] = readers[name][0]
    decision = read_outcome(gaps, lower, plan['decision_rule']['minimum_gap_points'] / 100)
    decision.update(primary=primary, intervals=intervals, reader_intervals=readers)

    status('SECONDARY bins, releases, reader counts and pools')
    counts, draws = plan['sentence_prediction']['reader_counts_for_curve'], plan['sentence_prediction']['subset_draws_count']
    secondary = {'by_release': {name: scored.by_release(v) for name, v in values.items()},
                 'reader_curve': {name: scored.curve(arms[name], counts, draws, seed) for name in ('real', 'gaussian')},
                 'within_block_ways': {name: {f'{w}_way': float(np.mean([within_score(scored.average(arms[name])[b], scored.candidates[b], w).mean() * len(b)
                                                                          for b in scored.blocks]) * len(scored.blocks) / len(scored.order)) for w in (2, 4)}
                                       for name in ('real', 'gaussian')},
                 'all_sentence_pool': {name: scored.across(scored.average(arms[name])) for name in ('real', 'gaussian')},
                 'bins': {}}
    for b in range(waves.shape[1] // channels):
        columns = waves[:, b * channels:(b + 1) * channels]
        entry = {}
        for name, options in (('real', features(columns)), ('gaussian', noise_options(data, columns, seed))):
            _, predicted, _ = run_arm(data, options, true, alphas, seed)
            entry[name] = float(scored.within(scored.average(predicted[0])).mean())
        secondary['bins'][f'{50 * b}_{50 * (b + 1)}_ms'] = entry
        status(f'Bin {b}: {entry}')
    return {'decision': decision, 'secondary': secondary, 'selection': selection, 'values': values}


__all__ = ['FIRST', 'SECOND', 'THIRD', 'read_outcome', 'run_test', 'with_gaze', 'without_channels']
