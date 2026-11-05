"""SYNTHETIC fixtures only. These test the pilot's software, not EEG performance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from eeg_decoding.data.dataset import Record
from eeg_decoding.evaluation.sentence_holdout import (apply_gate, bootstrap_gap, report_scores, score_queries,
                                                       sentence_groups, sentence_holdout_split)
from eeg_decoding.training.alignment_train import TextCache

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot', ROOT/'scripts/run_sentence_holdout_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def sentence(index, length=None):
    length = length or 6 + index % 12
    return " ".join(f"w{index}x{position}" for position in range(length))


def make_records(tmp_path, per_release=80, readers=3, shared=4, samples=240):
    rng = np.random.default_rng(5)
    rows = []
    for release, offset in (("zuco1", 0), ("zuco2", 1000)):
        texts = [sentence(offset+i) for i in range(per_release)] + [sentence(5000+i) for i in range(shared)]
        for reader in range(readers):
            for i, text in enumerate(texts):
                path = tmp_path/f"{release}_{reader}_{i}.npz"
                np.savez_compressed(path, eeg=rng.normal(size=(3, samples+i)).astype(np.float32))
                rows.append(Record(release, f"{release[-1]}R{reader}", "1", "nr", str(i), text, str(path),
                                   ["C3", "CZ", "C4"], 200., provenance={"units": "uV", "synthetic": True}))
    return rows


def make_text(rows, tmp_path, dimension=16):
    keys = sorted({r.key for r in rows})
    vectors = np.random.default_rng(3).normal(size=(len(keys), dimension)).astype(np.float32)
    path = tmp_path/"SYNTHETIC_text.npz"
    np.savez_compressed(path, keys=np.array(keys), embeddings=vectors, metadata=json.dumps({"synthetic": True}))
    return TextCache(path), path


def test_split_keeps_sentences_apart_and_readers_shared(tmp_path):
    rows = make_records(tmp_path)
    splits = sentence_holdout_split(rows)
    keys = {name: {r.key for r in part} for name, part in splits.items()}
    assert not keys["train"] & keys["val"] and not keys["train"] & keys["test"] and not keys["val"] & keys["test"]
    assert {r.subject_key for r in splits["train"]} == {r.subject_key for r in splits["test"]}
    assert sum(len(part) for part in splits.values()) == len(rows)
    both = {r.key for r in rows if r.dataset == "zuco1"} & {r.key for r in rows if r.dataset == "zuco2"}
    assert len(both) == 4 and both <= keys["train"]
    for name in ("val", "test"):
        assert abs(len({r.key for r in splits[name] if r.dataset == "zuco1"}) - 12) <= 1


def test_split_is_deterministic(tmp_path):
    rows = make_records(tmp_path)
    assert [r.key for r in sentence_holdout_split(rows)["test"]] == [r.key for r in sentence_holdout_split(rows)["test"]]


def test_near_duplicates_stay_together(tmp_path):
    rows = make_records(tmp_path, per_release=60, readers=1, shared=0)
    template = rows[0]
    variant = Record(template.dataset, template.subject, "1", "nr", "dup", template.text.replace(" ", " - ", 1),
                     template.signal_path, template.channels, template.sfreq, provenance=dict(template.provenance))
    assert variant.key != template.key
    groups, merged = sentence_groups(rows+[variant])
    assert len(merged) == 1
    splits = sentence_holdout_split(rows+[variant])
    where = [name for name, part in splits.items() if {template.key, variant.key} & {r.key for r in part}]
    assert len(where) == 1


def test_candidates_come_only_from_the_query_release(tmp_path):
    rows = make_records(tmp_path, per_release=30, readers=2, shared=0)
    text, _ = make_text(rows, tmp_path)
    perfect = text.get([r.key for r in rows])
    scored = score_queries(perfect, rows, text, ways=(2, 10))
    assert (scored["rank"] == 1).all() and (scored["expected"][10] == 1).all() and (scored["matched"] == 1).all()
    # Aim every zuco1 query at a zuco2 sentence: the zuco1 pool must ignore that candidate.
    foreign = next(r.key for r in rows if r.dataset == "zuco2")
    predicted = text.get([foreign if r.dataset == "zuco1" else r.key for r in rows])
    other = score_queries(predicted, rows, text, ways=(2, 10))
    assert all(rank <= 30 for rank in other["rank"])


def test_constant_prediction_averages_to_chance(tmp_path):
    rows = make_records(tmp_path, per_release=40, readers=1, shared=0)
    text, _ = make_text(rows, tmp_path)
    predicted = np.tile(np.random.default_rng(1).normal(size=16), (len(rows), 1)).astype(np.float32)
    scored = score_queries(predicted, rows, text, ways=(2, 10))
    report = report_scores(scored, rows)
    assert report["metrics"]["10_way"] == pytest.approx(.1, abs=1e-9)
    assert report["metrics"]["2_way"] == pytest.approx(.5, abs=1e-9)


def test_matched_pool_counts_a_tie_as_a_miss(tmp_path):
    rows = make_records(tmp_path, per_release=30, readers=1, shared=0)
    text, _ = make_text(rows, tmp_path)
    predicted = np.ones((len(rows), 16), dtype=np.float32)
    keys = sorted({r.key for r in rows})
    tied = TextCache.__new__(TextCache)
    tied.get = lambda wanted: np.ones((len(wanted), 16), dtype=np.float32)
    scored = score_queries(predicted, rows, tied, ways=(2,))
    assert len(keys) == 60 and (scored["matched"] == 0).all() and (scored["expected"][2] == 0).all()


def gate_rows(readers=30, sentences=40):
    class Row:
        def __init__(self, reader, key):
            self.subject_key, self.key = reader, key
    return [Row(f"r{a}", f"s{b}") for a in range(readers) for b in range(sentences)]


def test_gate_passes_only_when_all_conditions_hold():
    rows = gate_rows()
    rng = np.random.default_rng(0)
    base = rng.uniform(.05, .15, len(rows))
    strong = base + .05 + rng.normal(0, .01, len(rows))
    passed = apply_gate(strong, {"gaussian": base, "class_shuffled": base, "timing_only_ridge": base}, rows, iterations=300)
    assert passed["passed"] and passed["readers_positive"] == 30
    timing_wins = apply_gate(strong, {"gaussian": base, "class_shuffled": base, "timing_only_ridge": strong}, rows, iterations=300)
    assert not timing_wins["passed"] and not timing_wins["checks"]["gaps_at_least_minimum"]
    tiny = apply_gate(base+.001, {"gaussian": base, "class_shuffled": base, "timing_only_ridge": base}, rows, iterations=300)
    assert not tiny["passed"] and tiny["checks"]["readers_positive_against_gaussian"]


def test_gate_requires_most_readers_to_gain():
    rows = gate_rows()
    gain = np.array([.2 if int(r.subject_key[1:]) < 10 else -.01 for r in rows])
    base = np.full(len(rows), .1)
    result = apply_gate(base+gain, {"gaussian": base, "class_shuffled": base, "timing_only_ridge": base}, rows, iterations=300)
    assert result["gaps"]["gaussian"] >= .02 and result["readers_positive"] == 10 and not result["passed"]


def test_bootstrap_interval_excludes_zero_only_for_a_real_gap():
    rows = gate_rows()
    rng = np.random.default_rng(1)
    assert bootstrap_gap(.05+rng.normal(0, .01, len(rows)), rows, 300)["reader_ci95"][0] > 0
    centered = bootstrap_gap(rng.normal(0, .1, len(rows)), rows, 300)
    assert centered["reader_ci95"][0] < 0 < centered["reader_ci95"][1]


PLAN = {"seed": 42, "split": {"fractions": {"test": .15}},
        "models": {"ridge": {"alphas": [1., 100.]}, "cnn": {"max_hours_per_arm": 1}}}


def pilot_setup(tmp_path, config):
    rows = make_records(tmp_path, per_release=70, readers=2, shared=2)
    _, text_path = make_text(rows, tmp_path)
    cfg = dict(config, batch_size=4, gradient_accumulation_steps=1, epochs=1, patience=2, temporal_segments=1,
               alignment_candidate_scope="batch", embedding_dim=8, max_samples=240)
    return rows, text_path, cfg


def test_failed_gate_never_reads_test_signals(tmp_path, config, monkeypatch):
    rows, text_path, cfg = pilot_setup(tmp_path, config)
    held_out = {r.signal_path for r in sentence_holdout_split(rows)["test"]}
    opened = []
    from eeg_decoding import workflow
    original = workflow.load_signal
    monkeypatch.setattr(workflow, "load_signal", lambda r: (opened.append(r.signal_path), original(r))[1])
    monkeypatch.setattr(pilot, "apply_gate", lambda *a, **k: {"passed": False})
    output = tmp_path/"run"
    reports, decision, tests = pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10), readers_positive=1, iterations=50)
    assert tests == {} and not list(output.rglob("test.json"))
    assert opened and not set(opened) & held_out
    assert set(reports) == {"timing", "ridge_real", "ridge_gaussian", "ridge_shuffled", "cnn_real", "cnn_gaussian", "cnn_shuffled"}
    for report in reports.values():
        assert report["evaluation_split"] == "val" and report["n_queries"] > 0
    assert (output/"decision.json").is_file()
    with pytest.raises(FileExistsError):
        pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10))


def test_passing_gate_scores_the_test_sentences_once(tmp_path, config, monkeypatch):
    rows, text_path, cfg = pilot_setup(tmp_path, config)
    calls = iter([{"passed": True}, {"passed": False}])
    monkeypatch.setattr(pilot, "apply_gate", lambda *a, **k: next(calls))
    output = tmp_path/"run"
    _, _, tests = pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10), iterations=50)
    assert set(tests) == {"timing", "ridge_real", "ridge_gaussian", "ridge_shuffled"}
    assert (output/"ridge_real/test.json").is_file() and not (output/"cnn_real/test.json").exists()
    monkeypatch.setattr(pilot, "apply_gate", lambda *a, **k: {"passed": True})
    with pytest.raises(FileExistsError):
        pilot.run_pilot(rows, text_path, cfg, PLAN, output, ways=(2, 10), iterations=50, resume=True)


def test_both_passing_models_are_each_scored_once(tmp_path, config, monkeypatch):
    rows, text_path, cfg = pilot_setup(tmp_path, config)
    monkeypatch.setattr(pilot, "apply_gate", lambda *a, **k: {"passed": True})
    _, _, tests = pilot.run_pilot(rows, text_path, cfg, PLAN, tmp_path/"run", ways=(2, 10), iterations=50)
    assert set(tests) == {"timing", "ridge_real", "ridge_gaussian", "ridge_shuffled",
                          "cnn_real", "cnn_gaussian", "cnn_shuffled"}


def test_time_limit_stops_an_arm():
    wrapped = pilot.TimedLoader([1, 2], deadline=-1.)
    assert len(wrapped) == 2
    with pytest.raises(pilot.TimeLimitExceeded):
        list(wrapped)


def test_timing_features_contain_no_amplitudes(tmp_path):
    rows = make_records(tmp_path, per_release=10, readers=1, shared=0)
    first = pilot.timing_features(rows)
    for r in rows:
        with np.load(r.signal_path) as f:
            x = f["eeg"]
        np.savez_compressed(r.signal_path, eeg=x*0+1)
    np.testing.assert_array_equal(first, pilot.timing_features(rows))
