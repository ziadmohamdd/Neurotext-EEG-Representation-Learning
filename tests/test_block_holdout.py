"""SYNTHETIC fixtures only. These test the pilot's software, not EEG performance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from eeg_decoding.data.dataset import Record
from eeg_decoding.data.pipeline_view import PipelineView, ReaderNormalizer, center_crop, clean_signal, pipeline_loader
from eeg_decoding.evaluation.block_holdout import (apply_block_gate, block_holdout_split, bootstrap_gap, leakage_audit, list_ranges,
                                                    position_features, row_block, within_block_scores)
from eeg_decoding.training.alignment_train import TextCache

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_v2', ROOT/'scripts/run_block_holdout_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
PIPELINE = {"notch_hz": 50, "drop_channels": ["Cz"], "window_samples": 240}


def sentence(index):
    return " ".join(f"w{index}x{position}" for position in range(6 + index % 12))


def make_records(tmp_path, per_release=100, readers=2, shared=4, samples=240):
    rng = np.random.default_rng(5)
    rows = []
    for release, offset in (("zuco1", 0), ("zuco2", 1000)):
        # Shared sentences open each list, as they do near the start of the real lists.
        texts = [sentence(5000+i) for i in range(shared)] + [sentence(offset+i) for i in range(per_release)]
        for reader in range(readers):
            for i, text in enumerate(texts):
                path = tmp_path/f"{release}_{reader}_{i}.npz"
                x = rng.normal(size=(3, samples+i)).astype(np.float32)
                x[1] = 0
                np.savez_compressed(path, eeg=x)
                rows.append(Record(release, f"{release[-1]}R{reader}", "1", "nr", str(i), text, str(path),
                                   ["C3", "Cz", "C4"], 200., provenance={"units": "uV", "synthetic": True}))
    return rows


def make_text(rows, tmp_path, dimension=16):
    keys = sorted({r.key for r in rows})
    vectors = np.random.default_rng(3).normal(size=(len(keys), dimension)).astype(np.float32)
    path = tmp_path/"SYNTHETIC_text.npz"
    np.savez_compressed(path, keys=np.array(keys), embeddings=vectors, metadata=json.dumps({"synthetic": True}))
    return TextCache(path), path


def split(rows, seed=42):
    return block_holdout_split(rows, seed, block=12, buffer=3)


def test_blocks_are_contiguous_and_buffered(tmp_path):
    rows = make_records(tmp_path)
    splits, blocks, audit = split(rows)
    assert audit["buffer_observations"] > 0
    for release in ("zuco1", "zuco2"):
        held = {}
        for name in ("val", "test"):
            for r in splits[name]:
                if r.dataset == release:
                    held.setdefault(row_block(blocks, r), set()).add(int(r.sentence_id))
        assert len(held) == 4
        train = {int(r.sentence_id) for r in splits["train"] if r.dataset == release}
        for ids in held.values():
            assert len(ids) == 12 and max(ids) - min(ids) == 11
            # Nothing within the buffer distance of a block may be trained on.
            near = set(range(min(ids)-3, max(ids)+4))
            assert not near & train


def test_two_blocks_each_for_validation_and_test(tmp_path):
    rows = make_records(tmp_path)
    splits, blocks, _ = split(rows)
    for name in ("val", "test"):
        for release in ("zuco1", "zuco2"):
            names = {row_block(blocks, r) for r in splits[name] if r.dataset == release}
            assert len(names) == 2 and all(n.startswith(name) for n in names)


def test_shared_sentences_never_enter_held_out_blocks(tmp_path):
    rows = make_records(tmp_path)
    splits, _, _ = split(rows)
    both = {r.key for r in rows if r.dataset == "zuco1"} & {r.key for r in rows if r.dataset == "zuco2"}
    held = {r.key for name in ("val", "test") for r in splits[name]}
    assert len(both) == 4 and not both & held


def test_near_duplicates_never_enter_held_out_blocks(tmp_path):
    rows = make_records(tmp_path, readers=1, shared=0)
    splits, _, _ = split(rows)
    target = splits["val"][0]
    identity = 900
    copy = Record(target.dataset, target.subject, "1", "nr", str(identity), target.text.replace(" ", " - ", 1),
                  target.signal_path, target.channels, target.sfreq, provenance=dict(target.provenance))
    assert copy.key != target.key
    result, _, _ = split(rows+[copy])
    held = {r.key for name in ("val", "test") for r in result[name]}
    assert target.key not in held and copy.key not in held


def test_sentences_repeated_in_a_list_never_enter_held_out_blocks(tmp_path):
    rows = make_records(tmp_path, readers=1, shared=0)
    source = next(r for r in rows if r.dataset == "zuco2" and r.sentence_id == "50")
    repeat = Record(source.dataset, source.subject, "1", "nr", "700", source.text, source.signal_path,
                    source.channels, source.sfreq, provenance=dict(source.provenance))
    result, _, _ = split(rows+[repeat])
    assert source.key not in {r.key for name in ("val", "test") for r in result[name]}


def test_split_is_deterministic(tmp_path):
    rows = make_records(tmp_path)
    assert [r.signal_path for r in split(rows)[0]["test"]] == [r.signal_path for r in split(rows)[0]["test"]]


def test_notch_removes_mains_and_leaves_other_frequencies(tmp_path):
    t = np.arange(800) / 200
    x = np.stack([np.sin(2*np.pi*50*t) + np.sin(2*np.pi*10*t), np.zeros_like(t), np.sin(2*np.pi*10*t)]).astype(np.float32)
    path = tmp_path/"s.npz"
    np.savez_compressed(path, eeg=x)
    record = Record("zuco2", "R", "1", "nr", "0", "a b", str(path), ["C3", "Cz", "C4"], 200.)
    y = clean_signal(record, PIPELINE)
    assert y.shape[0] == 2
    middle = slice(200, 600)
    power = np.abs(np.fft.rfft(y[0, middle]))
    frequency = np.fft.rfftfreq(400, 1/200)
    assert power[np.argmin(np.abs(frequency-50))] < .05 * power[np.argmin(np.abs(frequency-10))]
    np.testing.assert_allclose(y[1, middle], x[2, middle], atol=.02)


def test_center_crop_keeps_shorter_signals():
    x = np.arange(20.).reshape(2, 10)
    assert center_crop(x, 20).shape[1] == 10
    np.testing.assert_array_equal(center_crop(x, 4)[0], [3, 4, 5, 6])


def test_per_reader_normalizer_uses_only_training_rows_and_refuses_new_readers(tmp_path):
    rows = make_records(tmp_path, per_release=6, readers=2, shared=0)
    reader = rows[0].subject_key
    fit = [r for r in rows if r.subject_key == reader][:4]
    norm = ReaderNormalizer().fit(fit, PIPELINE, per_reader=True)
    x = clean_signal(fit[0], PIPELINE)
    joined = np.concatenate([clean_signal(r, PIPELINE) for r in fit], axis=1)
    out = norm.transform(joined, reader)
    np.testing.assert_allclose(out.mean(1), 0, atol=1e-5)
    np.testing.assert_allclose(out.std(1), 1, atol=1e-5)
    with pytest.raises(ValueError, match="no training statistics"):
        norm.transform(x, "someone else")


def test_shared_normalizer_accepts_any_reader(tmp_path):
    rows = make_records(tmp_path, per_release=6, readers=2, shared=0)
    norm = ReaderNormalizer().fit(rows[:6], PIPELINE, per_reader=False)
    assert norm.transform(clean_signal(rows[0], PIPELINE), "someone else").shape[0] == 2


def test_view_matches_controls(tmp_path):
    rows = make_records(tmp_path, per_release=6, readers=1, shared=0)
    norm = ReaderNormalizer().fit(rows, PIPELINE, per_reader=True)
    real = PipelineView(rows, norm, PIPELINE)[0]["eeg"]
    noise = PipelineView(rows, norm, PIPELINE, "gaussian")[0]["eeg"]
    assert real.shape == noise.shape and real.shape[0] == 2 and real.shape[1] == 240
    assert not np.allclose(real.numpy(), noise.numpy())
    keys = PipelineView(rows, norm, PIPELINE, "shuffled").training_keys
    assert all(a != r.key for a, r in zip(keys, rows))


def test_within_block_score_uses_only_the_own_block(tmp_path):
    rows = make_records(tmp_path)
    splits, blocks, _ = split(rows)
    text, _ = make_text(rows, tmp_path)
    val = splits["val"]
    perfect = text.get([r.key for r in val])
    assert (within_block_scores(perfect, val, blocks, text) == 1).all()
    constant = np.tile(np.random.default_rng(1).normal(size=16), (len(val), 1)).astype(np.float32)
    assert within_block_scores(constant, val, blocks, text).mean() == pytest.approx(.1, abs=1e-9)
    # Aim every query at a sentence from another block: its own pool must not see that candidate.
    names = sorted({row_block(blocks, r) for r in val})
    foreign = next(r.key for r in val if row_block(blocks, r) == names[-1])
    moved = text.get([foreign for _ in val])
    assert within_block_scores(moved, val, blocks, text).max() <= 1


def test_position_features_contain_no_signal_and_scale_with_terms(tmp_path):
    rows = make_records(tmp_path, per_release=10, readers=1, shared=0)
    ranges = list_ranges(rows)
    x = position_features(rows, 3, ranges)
    assert x.shape == (len(rows), 7)
    np.testing.assert_array_equal(x[0, 1:], x[10, 1:])
    assert x[0, 0] == 0 and x[10, 0] == 1


def gate_rows(readers=30, sentences=40):
    class Row:
        def __init__(self, reader, key):
            self.subject_key, self.key = reader, key
    return [Row(f"r{a}", f"s{b}") for a in range(readers) for b in range(sentences)]


def comparators(base, **changes):
    return dict({"gaussian": base, "class_shuffled": base, "timing_only_ridge": base, "position_only_ridge": base}, **changes)


def test_gate_requires_every_condition():
    rows = gate_rows()
    rng = np.random.default_rng(0)
    base = rng.uniform(.05, .15, len(rows))
    strong = base + .05 + rng.normal(0, .01, len(rows))
    assert apply_block_gate(strong, comparators(base), strong, base, rows, iterations=300)["passed"]
    position = apply_block_gate(strong, comparators(base, position_only_ridge=strong), strong, base, rows, iterations=300)
    assert not position["passed"] and not position["checks"]["primary_gaps_at_least_minimum"]
    within = apply_block_gate(strong, comparators(base), base, base, rows, iterations=300)
    assert not within["passed"] and not within["checks"]["within_block_gap_at_least_minimum"]
    few = np.array([.2 if int(r.subject_key[1:]) < 10 else -.01 for r in rows]) + base
    assert not apply_block_gate(few, comparators(base), few, base, rows, iterations=300)["checks"]["readers_positive_against_gaussian"]


def test_bootstrap_level_widens_the_interval():
    rows = gate_rows()
    gap = np.random.default_rng(1).normal(.01, .1, len(rows))
    narrow = bootstrap_gap(gap, rows, 500, level=90)["reader_interval"]
    wide = bootstrap_gap(gap, rows, 500, level=98.33)["reader_interval"]
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


PLAN = {"seed": 42,
        "split": {"blocks_per_release": {"val": 2, "test": 2}, "block_sentences": 12, "buffer_sentences_each_side": 3},
        "pipelines": {"V": {"notch_hz": 50, "drop_channels": ["Cz"]}, "F": {"notch_hz": 50, "drop_channels": ["Cz"]}},
        "shared_controls": {"position_only_ridge": {"fourier_terms": [2], "alphas": [1.]}},
        "models": {"ridge": {"pipelines": ["V", "F"], "alphas": [1., 100.]},
                   "cnn": {"pipelines": ["F"], "model_seeds": [42, 43], "max_hours_per_arm": 1}}}


def setup(tmp_path, config):
    rows = make_records(tmp_path)
    _, text_path = make_text(rows, tmp_path)
    cfg = dict(config, batch_size=4, gradient_accumulation_steps=1, epochs=1, patience=2, temporal_segments=1,
               alignment_candidate_scope="batch", embedding_dim=8, window_samples=240)
    return rows, text_path, cfg


def test_failed_gate_never_reads_test_signals(tmp_path, config, monkeypatch):
    rows, text_path, cfg = setup(tmp_path, config)
    held_out = {r.signal_path for r in split(rows)[0]["test"]}
    opened = []
    from eeg_decoding.data import pipeline_view
    original = pipeline_view.load_signal
    monkeypatch.setattr(pipeline_view, "load_signal", lambda r: (opened.append(r.signal_path), original(r))[1])
    original_timing = pilot.timing_features
    monkeypatch.setattr(pilot, "timing_features", lambda part, counts: (opened.extend(r.signal_path for r in part), original_timing(part, counts))[1])
    monkeypatch.setattr(pilot, "apply_block_gate", lambda *a, **k: {"passed": False})
    output = tmp_path/"run"
    reports, decision, tests = pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10), iterations=50)
    assert tests == {} and not list(output.rglob("test.json"))
    assert opened and not set(opened) & held_out
    assert {"timing", "position", "ridge_V_real", "ridge_F_gaussian", "cnn_F_real_s42", "cnn_F_shuffled_s43"} <= set(reports)
    assert set(decision) == {"ridge_V", "ridge_F", "cnn_F"}
    with pytest.raises(FileExistsError):
        pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10))


def test_passing_gate_scores_test_blocks_once(tmp_path, config, monkeypatch):
    rows, text_path, cfg = setup(tmp_path, config)
    calls = iter([{"passed": False}, {"passed": False}, {"passed": True}])
    monkeypatch.setattr(pilot, "apply_block_gate", lambda *a, **k: next(calls))
    output = tmp_path/"run"
    _, _, tests = pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10), iterations=50)
    assert {"timing", "position", "cnn_F_real_s42", "cnn_F_real_s43", "cnn_F_gaussian_s42"} <= set(tests)
    assert not (output/"ridge_V_real/test.json").exists()
    monkeypatch.setattr(pilot, "apply_block_gate", lambda *a, **k: {"passed": True})
    with pytest.raises(FileExistsError):
        pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10), iterations=50, resume=True)


def test_leakage_audit_passes_a_clean_split_and_catches_each_leak(tmp_path):
    rows = make_records(tmp_path)
    text, _ = make_text(rows, tmp_path)
    splits, blocks, _ = split(rows)
    assert leakage_audit(splits, blocks, rows, text, 3, cosine=1.1)["failures"] == []
    # A held-out sentence copied into training.
    leaked = dict(splits, train=splits["train"]+[splits["val"][0]])
    assert any("appears in train and val" in f for f in leakage_audit(leaked, blocks, rows, text, 3, cosine=1.1)["failures"])
    # Training placed inside a block's buffer.
    r = splits["val"][0]
    ids = [int(x.sentence_id) for x in splits["val"] if row_block(blocks, x) == row_block(blocks, r)]
    near = Record(r.dataset, r.subject, "1", "nr", str(max(ids)+1), "fresh words only here now", r.signal_path,
                  r.channels, r.sfreq, provenance=dict(r.provenance))
    keys = sorted({x.key for x in rows})
    cache = TextCache.__new__(TextCache)
    table = dict(zip(keys, text.get(keys)))
    table[near.key] = np.random.default_rng(9).normal(size=16).astype(np.float32)
    cache.get = lambda wanted: np.stack([table[k] for k in wanted])
    found = leakage_audit(dict(splits, train=splits["train"]+[near]), blocks, rows+[near], cache, 3, cosine=1.1)["failures"]
    assert any("list positions" in f for f in found)
    # A held-out sentence lightly reworded, with a matching embedding, in training.
    copy = Record(r.dataset, r.subject, "1", "nr", "990", r.text+" extra", r.signal_path, r.channels, r.sfreq, provenance=dict(r.provenance))
    table[copy.key] = table[r.key]
    found = leakage_audit(dict(splits, train=splits["train"]+[copy]), blocks, rows+[copy], cache, 3)["failures"]
    assert any("Near-duplicate" in f for f in found) and any("nearly contains" in f for f in found) and any("cosine" in f for f in found)


def test_run_refuses_to_start_when_the_audit_fails(tmp_path, config, monkeypatch):
    rows, text_path, cfg = setup(tmp_path, config)
    monkeypatch.setattr(pilot, "leakage_audit", lambda *a, **k: {"failures": ["planted"]})
    with pytest.raises(ValueError, match="Leakage audit failed: planted"):
        pilot.run_pilot(rows, text_path, cfg, PLAN, tmp_path/"run", ways=(2, 10), iterations=50)
    assert not list((tmp_path/"run").rglob("*.pt"))


def test_changing_held_out_signals_cannot_change_fitted_statistics(tmp_path):
    rows = make_records(tmp_path, per_release=60, readers=1, shared=0)
    splits, _, _ = block_holdout_split(rows, 42, block=6, buffer=2)
    fit = lambda: ReaderNormalizer().fit(splits["train"], PIPELINE, per_reader=True).state_dict()
    first = fit()
    train_paths = {r.signal_path for r in splits["train"]}
    for name in ("val", "test"):
        for r in splits[name]:
            assert r.signal_path not in train_paths
            np.savez_compressed(r.signal_path, eeg=np.full((3, 240), 1000., dtype=np.float32))
    assert fit() == first


def test_time_limit_stops_an_arm():
    with pytest.raises(pilot.TimeLimitExceeded):
        list(pilot.TimedLoader([1, 2], deadline=-1.))


def test_loader_batches_have_the_reduced_channel_count(tmp_path):
    rows = make_records(tmp_path, per_release=8, readers=1, shared=0)
    norm = ReaderNormalizer().fit(rows, PIPELINE, per_reader=True)
    batch = next(iter(pipeline_loader(rows, dict(batch_size=4, pin_memory=False, seed=42), norm, PIPELINE, True, True)))
    assert batch["eeg"].shape[1] == 2 and batch["mask"].shape[0] == 4
