"""SYNTHETIC fixtures only. These test the attribution software, not EEG performance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from eeg_decoding.data.dataset import Record
from eeg_decoding.data.pipeline_view import ReaderNormalizer, clean_signal
from eeg_decoding.evaluation.proximity import (closest_text_prediction, excluded_positions, forbidden_positions, proximity_audit,
                                               proximity_decision, proximity_folds, rows_for)
from eeg_decoding.training.alignment_train import TextCache

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('attribution_v3', ROOT/'scripts/run_proximity_attribution.py')
attribution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attribution)
LAYOUT = dict(cycle=30, cluster=6, offsets=(0, 10, 20), near=2, gap=2)
ZONE = dict(block=8, buffer=3, val_blocks=1, test_blocks=1)


def sentence(index):
    return " ".join(f"w{index}x{position}" for position in range(6 + index % 12))


def make_records(tmp_path, per_release=200, readers=2, samples=240):
    rng = np.random.default_rng(5)
    rows = []
    for release, offset in (("zuco1", 0), ("zuco2", 1000)):
        for reader in range(readers):
            for i in range(per_release):
                path = tmp_path/f"{release}_{reader}_{i}.npz"
                x = rng.normal(size=(3, samples+i % 20)).astype(np.float32)
                x[1] = 0
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


def build(rows):
    excluded = excluded_positions(rows, 42, **ZONE)
    return excluded, proximity_folds(rows, excluded, **LAYOUT)


def audit(rows, folds, excluded, text, **kwargs):
    return proximity_audit(rows, folds, excluded, text, LAYOUT["near"], LAYOUT["gap"], ways=10, select_way=4, cosine=1.1, **kwargs)


def test_the_held_out_sets_are_the_same_for_near_and_far_and_never_repeat(tmp_path):
    rows = make_records(tmp_path, readers=1)
    _, folds = build(rows)
    for release in ("zuco1", "zuco2"):
        seen = set()
        for roles in folds:
            held = roles["select"][release] | roles["score"][release]
            assert not held & seen
            seen |= held


def test_near_and_far_match_in_size_and_differ_only_in_closeness(tmp_path):
    rows = make_records(tmp_path, readers=1)
    _, folds = build(rows)
    for roles in folds:
        for release in ("zuco1", "zuco2"):
            held = roles["select"][release] | roles["score"][release]
            assert len(roles["near"][release]) == len(roles["far"][release])
            assert len(roles["near_half"][release]) == len(roles["far_half"][release])
            assert min(min(abs(i-h) for h in held) for i in roles["far"][release]) > LAYOUT["near"] + LAYOUT["gap"]
            assert all(min(abs(i-h) for h in held) <= LAYOUT["near"] for i in roles["near_only"][release])
            assert roles["near_only"][release] <= roles["near"][release]
            # Everything in the near condition beyond the close sentences comes from the far set.
            assert roles["near"][release] - roles["near_only"][release] <= roles["far"][release]
            assert roles["near_half"][release] <= roles["near"][release] and roles["far_half"][release] <= roles["far"][release]


def test_the_gap_sentences_are_never_used(tmp_path):
    rows = make_records(tmp_path, readers=1)
    _, folds = build(rows)
    for roles in folds:
        for release in ("zuco1", "zuco2"):
            held = roles["select"][release] | roles["score"][release]
            used = roles["near"][release] | roles["far"][release]
            for i in used:
                assert not LAYOUT["near"] < min(abs(i-h) for h in held) <= LAYOUT["near"] + LAYOUT["gap"]


def test_the_v2_test_zone_is_in_no_role(tmp_path):
    rows = make_records(tmp_path, readers=1)
    excluded, folds = build(rows)
    assert excluded["zuco1"] and excluded["zuco2"]
    for roles in folds:
        for value in roles.values():
            for release, ids in value.items():
                assert not ids & excluded[release]


def test_shared_repeated_and_contained_sentences_are_forbidden(tmp_path):
    rows = make_records(tmp_path, per_release=30, readers=1)
    source = rows[3]
    shared = Record("zuco2", rows[-1].subject, "1", "nr", "500", source.text, source.signal_path, source.channels, source.sfreq, provenance=dict(source.provenance))
    longer = Record("zuco1", source.subject, "1", "nr", "501", rows[7].text + " plus many more words here indeed now", source.signal_path,
                    source.channels, source.sfreq, provenance=dict(source.provenance))
    found = forbidden_positions(rows + [shared, longer])
    assert 3 in found["zuco1"] and 500 in found["zuco2"]
    assert 7 in found["zuco1"] and 501 in found["zuco1"]


def test_folds_are_deterministic(tmp_path):
    rows = make_records(tmp_path, readers=1)
    assert build(rows)[1] == build(rows)[1]


def test_audit_passes_a_clean_split_and_catches_each_leak(tmp_path):
    rows = make_records(tmp_path)
    text, _ = make_text(rows, tmp_path)
    excluded, folds = build(rows)
    assert audit(rows, folds, excluded, text)["failures"] == []

    def changed(**roles):
        copy = [{name: {k: set(v) for k, v in value.items()} for name, value in fold.items()} for fold in folds]
        copy[0].update(roles)
        return copy
    held = next(iter(folds[0]["score"]["zuco1"]))
    far = dict(folds[0]["far"], zuco1=folds[0]["far"]["zuco1"] | {held})
    assert any("contains a held-out" in f for f in audit(rows, changed(far=far), excluded, text)["failures"])
    close = next(iter(folds[0]["near_only"]["zuco1"]))
    far = dict(folds[0]["far"], zuco1=folds[0]["far"]["zuco1"] | {close})
    assert any("far sentence" in f for f in audit(rows, changed(far=far), excluded, text)["failures"])
    zone = next(iter(excluded["zuco1"]))
    near = dict(folds[0]["near"], zuco1=folds[0]["near"]["zuco1"] | {zone})
    found = audit(rows, changed(near=near), excluded, text)["failures"]
    assert any("v2 test zone" in f for f in found) and any("differ in size" in f for f in found)
    score = dict(folds[0]["score"], zuco1=folds[0]["score"]["zuco1"] | {next(iter(folds[0]["select"]["zuco1"]))})
    assert any("selection sentence is scored" in f for f in audit(rows, changed(score=score), excluded, text)["failures"])


def test_closest_text_uses_only_the_given_training_sentences(tmp_path):
    rows = make_records(tmp_path, per_release=10, readers=1)
    lookup = {(r.dataset, int(r.sentence_id)): np.full(4, float(r.sentence_id)) for r in rows}
    query = [r for r in rows if r.dataset == "zuco1" and r.sentence_id == "5"]
    predicted = closest_text_prediction(query, {"zuco1": {2, 3, 9}, "zuco2": {1}}, lookup)
    # Ids 3 and 2 are the closest to 5 among the given training sentences.
    np.testing.assert_allclose(predicted[0], 2.5)


def decision_rows(readers=30, sentences=40):
    class Row:
        def __init__(self, reader, key):
            self.subject_key, self.key = reader, key
    return [Row(f"r{a}", f"s{b}") for a in range(readers) for b in range(sentences)]


@pytest.mark.parametrize("near_level, far_level, noise_level, label", [
    (.20, .10, .10, "Closeness explains most of the score"),
    (.13, .10, .10, "Closeness explains part of the score"),
    (.14, .14, .10, "Closeness is not the main cause"),
    (.10, .10, .10, "Inconclusive"),
    (.25, .15, .10, "Inconclusive"),
])
def test_decision_follows_the_frozen_table(near_level, far_level, noise_level, label):
    rows = decision_rows()
    noise = np.random.default_rng(0).normal(0, .002, len(rows))
    near, far = near_level + noise, far_level + noise
    assert proximity_decision(near, far, noise_level + noise, rows, iterations=200)["label"] == label


def test_decision_needs_a_clear_gain():
    rows = decision_rows()
    # A large mean gain that half the sentences contradict is not clear when sentences are resampled.
    far = np.full(len(rows), .14)
    near = far + np.array([.5 if int(r.key[1:]) % 2 else -.38 for r in rows])
    result = proximity_decision(near, far, far, rows, iterations=200)
    assert result["gain"] >= .05 and result["lower_bounds"]["sentence"] <= 0
    assert result["label"] == "Closeness is not the main cause"


PLAN = {"seed": 42,
        "split": {"v2_test_zone": ZONE, "cycle_sentences": 30, "cluster_sentences": 6, "fold_offsets": [0, 10, 20], "near_lag": 2, "gap_lags": 2},
        "inputs": {"F": {"notch_hz": 50, "drop_channels": ["Cz"]}, "v1": {"notch_hz": None, "drop_channels": []}},
        "runs": {"ridge": {"alphas": [1., 100.]},
                 "cnn": {"model_seed_by_fold": [42, 43, 44], "max_hours_per_arm": 1},
                 "no_eeg_controls": {"position": {"fourier_terms": [2], "alphas": [1.]}}}}


def setup(tmp_path, config):
    rows = make_records(tmp_path, per_release=150)
    _, text_path = make_text(rows, tmp_path)
    cfg = dict(config, batch_size=4, gradient_accumulation_steps=1, epochs=1, patience=2, temporal_segments=1,
               alignment_candidate_scope="batch", embedding_dim=8, window_samples=240)
    return rows, text_path, cfg


def test_run_never_opens_the_v2_test_zone_and_writes_a_decision(tmp_path, config, monkeypatch):
    rows, text_path, cfg = setup(tmp_path, config)
    excluded = excluded_positions(rows, 42, **ZONE)
    blocked = {r.signal_path for r in rows if int(r.sentence_id) in excluded[r.dataset]}
    opened = []
    from eeg_decoding.data import pipeline_view
    original = pipeline_view.load_signal
    monkeypatch.setattr(pipeline_view, "load_signal", lambda r: (opened.append(r.signal_path), original(r))[1])
    original_timing = attribution.timing_features
    monkeypatch.setattr(attribution, "timing_features", lambda part, counts: (opened.extend(r.signal_path for r in part), original_timing(part, counts))[1])
    monkeypatch.setattr(attribution, "ARMS", [("F", c, "full", a) for c in ("near", "far") for a in ("real", "gaussian", "shuffled")] +
                        [("F", c, "half", "real") for c in ("near", "far")] + [("v1", c, "full", "real") for c in ("near", "far")])
    output = tmp_path/"run"
    collected, decision = attribution.run_attribution(rows, text_path, cfg, PLAN, output, iterations=50)
    assert opened and not set(opened) & blocked
    assert {"cnn", "ridge", "no_eeg", "scored_observations"} <= set(decision)
    assert decision["cnn"]["label"] and "last_epoch" in decision["cnn"]
    assert set(decision["no_eeg"]) == {"timing", "position", "text"}
    assert (output/"fold2/cnn_v1_full_far_real/best.pt").exists()
    assert {"scored.json", "last.pt"} <= {p.name for p in (output/"fold0/cnn_F_full_near_real").iterdir()}
    scored = {tuple(q["identity"]) for f in range(3) for q in collected[f, "cnn_F_full_near_real"]["queries"]}
    assert len(scored) == decision["scored_observations"]
    with pytest.raises(FileExistsError):
        attribution.run_attribution(rows, text_path, cfg, PLAN, output)


def test_near_and_far_share_queries_and_differ_in_training(tmp_path, config):
    rows, text_path, cfg = setup(tmp_path, config)
    output = tmp_path/"run"
    collected, _ = attribution.run_attribution(rows, text_path, cfg, PLAN, output, iterations=50)
    near, far = collected[0, "ridge_F_full_near_real"], collected[0, "ridge_F_full_far_real"]
    assert [q["identity"] for q in near["queries"]] == [q["identity"] for q in far["queries"]]
    roles = json.loads((output/"fold0/roles.json").read_text(encoding="utf-8"))
    assert roles["near"]["zuco1"] != roles["far"]["zuco1"]


def test_run_refuses_to_start_when_the_audit_fails(tmp_path, config, monkeypatch):
    rows, text_path, cfg = setup(tmp_path, config)
    monkeypatch.setattr(attribution, "proximity_audit", lambda *a, **k: {"failures": ["planted"]})
    with pytest.raises(ValueError, match="Leakage audit failed: planted"):
        attribution.run_attribution(rows, text_path, cfg, PLAN, tmp_path/"run", iterations=50)
    assert not list((tmp_path/"run").rglob("*.pt"))


def test_the_v1_input_keeps_every_channel_and_the_f_input_drops_cz(tmp_path):
    rows = make_records(tmp_path, per_release=6, readers=1)
    assert clean_signal(rows[0], PLAN["inputs"]["v1"]).shape[0] == 3
    assert clean_signal(rows[0], PLAN["inputs"]["F"]).shape[0] == 2
    np.testing.assert_array_equal(clean_signal(rows[0], PLAN["inputs"]["v1"]), np.load(rows[0].signal_path)["eeg"])


def test_changing_held_out_signals_cannot_change_fitted_statistics(tmp_path):
    rows = make_records(tmp_path, readers=1)
    _, folds = build(rows)
    train = rows_for(rows, folds[0]["near"])
    pipeline = PLAN["inputs"]["F"]
    first = ReaderNormalizer().fit(train, pipeline, per_reader=True).state_dict()
    for r in rows_for(rows, folds[0]["score"]):
        np.savez_compressed(r.signal_path, eeg=np.full((3, 240), 1000., dtype=np.float32))
    assert ReaderNormalizer().fit(train, pipeline, per_reader=True).state_dict() == first


def test_time_limit_stops_an_arm():
    with pytest.raises(attribution.TimeLimitExceeded):
        list(attribution.TimedLoader([1, 2], deadline=-1.))
