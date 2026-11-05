"""SYNTHETIC fixtures only. These test the bridge software, not EEG performance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from eeg_decoding.data.dataset import Record
from eeg_decoding.evaluation.position_bridge import (bridge_prediction, bridge_reading, bridge_scope, fit_positions, identity, missing_runs,
                                                     predict_positions, random_split_baseline, scope_audit, shuffled_position_map)
from eeg_decoding.evaluation.proximity import excluded_positions
from eeg_decoding.training.alignment_train import TextCache

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bridge_v4', ROOT/'scripts/run_position_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
ZONE = dict(block=8, buffer=3, val_blocks=1, test_blocks=1)
PLAN = {'seed': 42, 'protected': {'v2_test_zone': ZONE}, 'reference': {'minimum_identical_rank_share': .99},
        'bridge': {'alphas': [1, 100], 'neighbours': 2, 'shuffles': 2},
        'evaluation': {'primary_input': 'v1'}, 'random_splits': {'fractions': [.15], 'seeds': 2},
        'inputs': {'v1': {'notch_hz': None, 'drop_channels': []}}}


def sentence(index):
    return " ".join(f"w{index}x{position}" for position in range(6 + index % 12))


def make_records(tmp_path, per_release=200, readers=2, samples=240, drift=0.):
    rng = np.random.default_rng(5)
    rows = []
    for release, offset in (("zuco1", 0), ("zuco2", 1000)):
        for reader in range(readers):
            for i in range(per_release):
                path = tmp_path/f"{release}_{reader}_{i}.npz"
                x = rng.normal(size=(3, samples)).astype(np.float32)
                x[0] *= 1 + drift * i / per_release
                np.savez_compressed(path, eeg=x)
                rows.append(Record(release, f"{release[-1]}R{reader}", "1", "nr", str(i), sentence(offset+i), str(path),
                                   ["C3", "Cz", "C4"], 200., provenance={"units": "uV", "synthetic": True}))
    return rows


def make_text(rows, tmp_path, dimension=16):
    keys = sorted({r.key for r in rows})
    vectors = np.random.default_rng(3).normal(size=(len(keys), dimension)).astype(np.float32)
    path = tmp_path/"SYNTHETIC_text.npz"
    np.savez_compressed(path, keys=np.array(keys), embeddings=vectors, metadata=json.dumps({"synthetic": True}))
    return TextCache(path), path


def test_scope_removes_every_protected_sentence(tmp_path):
    rows = make_records(tmp_path, 120)
    zone = excluded_positions(rows, 42, **ZONE)
    train, val, closed, v1_test = bridge_scope(rows, 42, zone)
    used = {identity(r) for r in train + val}
    assert closed and v1_test
    assert not used & closed and not used & v1_test
    assert scope_audit(train, val, closed, v1_test)['failures'] == []


def test_audit_names_each_breach(tmp_path):
    rows = make_records(tmp_path, 120)
    zone = excluded_positions(rows, 42, **ZONE)
    train, val, closed, v1_test = bridge_scope(rows, 42, zone)
    assert any('share an ID' in f for f in scope_audit(train + val[:1], val, closed, v1_test)['failures'])
    breach = [r for r in rows if identity(r) in closed][:1]
    assert any('test-zone' in f for f in scope_audit(train + breach, val, closed, v1_test)['failures'])
    breach = [r for r in rows if identity(r) in v1_test and identity(r) not in closed][:1]
    assert any('v1 test' in f for f in scope_audit(train + breach, val, closed, v1_test)['failures'])


def test_position_shuffle_is_per_sentence_and_within_release(tmp_path):
    rows = make_records(tmp_path, 40)
    mapping = shuffled_position_map(rows, 1)
    assert sorted(v for (d, _), v in mapping.items() if d == 'zuco1') == [float(i) for i in range(40)]
    assert any(float(i) != v for (_, i), v in mapping.items())


def test_position_model_recovers_a_synthetic_drift_and_shuffle_does_not(tmp_path):
    rows = make_records(tmp_path, 60, readers=3)
    train = [r for r in rows if int(r.sentence_id) % 5]
    test = [r for r in rows if not int(r.sentence_id) % 5]
    make = lambda items: np.array([[float(r.sentence_id) + np.random.default_rng(int(r.sentence_id)).normal(), 1.] for r in items])
    models, _ = fit_positions(make(train), train, [1, 100], 42)
    estimate = predict_positions(models, make(test), test)
    truth = np.array([float(r.sentence_id) for r in test])
    assert np.mean(np.abs(estimate - truth)) < 2
    models, _ = fit_positions(make(train), train, [1, 100], 42, shuffled_position_map(train, 7))
    assert np.mean(np.abs(predict_positions(models, make(test), test) - truth)) > 5


def test_bridge_only_uses_training_text(tmp_path):
    rows = make_records(tmp_path, 20, readers=1)
    train_ids = {'zuco1': {2, 8}, 'zuco2': {3, 9}}
    lookup = {('zuco1', 2): np.array([1., 0.]), ('zuco1', 8): np.array([0., 1.]), ('zuco2', 3): np.array([1., 1.]), ('zuco2', 9): np.array([0., 0.])}
    query = [r for r in rows if r.dataset == 'zuco1'][:1]
    assert bridge_prediction(np.array([3.]), query, train_ids, lookup, 1)[0].tolist() == [1., 0.]
    assert bridge_prediction(np.array([3.]), query, train_ids, lookup, 2)[0].tolist() == [.5, .5]
    # A tie goes to the earlier sentence.
    assert bridge_prediction(np.array([5.]), query, train_ids, lookup, 1)[0].tolist() == [1., 0.]


def test_reading_labels(tmp_path):
    rows = make_records(tmp_path, 30, readers=4)
    rng = np.random.default_rng(0)
    base = rng.uniform(0, .1, len(rows))
    high = base + .3
    assert bridge_reading(high, base, .3, rows, 200)['label'] == 'Position alone could account for the CNN score'
    assert bridge_reading(high, base, .9, rows, 200)['label'] == 'Position explains part of the CNN score'
    assert bridge_reading(base, base, .1, rows, 200)['label'] == 'This route is not supported'
    # Reaching the reference without beating shuffled is still not support.
    assert bridge_reading(base, base, 0., rows, 200)['beats_shuffled'] is False


def test_missing_runs_finds_contiguous_gaps(tmp_path):
    rows = make_records(tmp_path, 12, readers=2)
    kept = [r for r in rows if not (r.subject == '1R1' and r.dataset == 'zuco1' and 3 <= int(r.sentence_id) <= 6)]
    runs = missing_runs(kept)
    assert runs['zuco1:1R1'] == [[3, 6]]
    assert runs['zuco1:1R0'] == []


def test_random_baseline_never_uses_protected_sentences(tmp_path, monkeypatch):
    rows = make_records(tmp_path, 120)
    text, _ = make_text(rows, tmp_path)
    zone = excluded_positions(rows, 42, **ZONE)
    _, _, closed, v1_test = bridge_scope(rows, 42, zone)
    lookup = {identity(r): text.get([r.key])[0] for r in rows if identity(r) not in closed and identity(r) not in v1_test}
    # A protected sentence would raise a KeyError on this restricted lookup.
    result = random_split_baseline(rows, closed, v1_test, lookup, text, [.15], range(2), ways=(2, 4))
    assert result['0.15']['splits'] == 2 and 0 <= result['0.15']['4_way']['mean'] <= 1


def test_runner_end_to_end_and_protects_scope(tmp_path):
    rows = make_records(tmp_path, 120, drift=3.)
    text, text_path = make_text(rows, tmp_path)
    output = tmp_path/'out'
    output.mkdir()
    seen = []

    def reference(items):
        seen.extend(items)
        return np.random.default_rng(0).normal(size=(len(items), 16)).astype(np.float32)

    plan = json.loads(json.dumps(PLAN))
    bridge.WAYS, bridge.MATCHED, old = (2, 4), 4, (bridge.WAYS, bridge.MATCHED, bridge.PRIMARY)
    bridge.PRIMARY = 4
    try:
        results, decision, splits = bridge.run_bridge(rows, text_path, plan, output, reference=reference, verify=lambda *a: {'share_identical_rank': 1.},
                                                      iterations=50, fractions=[.15], seeds=range(1), zone=ZONE, status=lambda m: None)
    finally:
        bridge.WAYS, bridge.MATCHED, bridge.PRIMARY = old
    zone = excluded_positions(rows, 42, **ZONE)
    _, _, closed, v1_test = bridge_scope(rows, 42, zone)
    assert not {identity(r) for r in seen} & (closed | v1_test)
    assert {'v1_cnn', 'neighbour_text', 'bridge_v1_real', 'bridge_v1_gaussian', 'bridge_v1_shuffled'} <= set(results)
    assert decision['label'] in ('Position alone could account for the CNN score', 'Position explains part of the CNN score', 'This route is not supported')
    assert (output/'decision.json').exists() and (output/'sentence_id_check.json').exists()


def test_main_refuses_a_changed_plan(tmp_path, monkeypatch):
    changed = tmp_path/'plan.json'
    changed.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(bridge, 'PLAN', changed)
    with pytest.raises(ValueError, match='changed'):
        bridge.main(tmp_path/'output')
