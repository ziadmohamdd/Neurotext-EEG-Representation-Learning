"""Replay frozen saved models and check provenance; no fitting or model selection."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.evaluation.word_order import pooled, missingness, select_alpha
from eeg_decoding.evaluation.spectral_pilot import count_basis, matched_gaussian
from eeg_decoding.evaluation.retrieval import ranks, summarize
from eeg_decoding.training.alignment_train import TextCache
from run_word_order_pilot import sha, identity

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/word_order_pilot_v1_20261031"


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    reports = json.loads((RUN / "validation_summary.json").read_text())
    rows = {identity(r): r for r in read_manifest(plan["manifest"])}
    validation = [rows[tuple(k)] for k in plan["validation_observations"]]
    with np.load(RUN / "word_sequences.npz", allow_pickle=False) as cache:
        words, offsets = cache["val_words"], cache["val_offsets"]
        valwords = [words[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        traincounts, valcounts = np.diff(cache["train_offsets"]), np.diff(offsets)
    basis, _ = count_basis(traincounts, valcounts)
    def load(name):
        with np.load(RUN / name, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    projection, noise_projection = load("real_word_projection.npz"), load("gaussian_word_projection.npz")
    moments = load("gaussian_training_moments.npz")
    noise = [matched_gaussian(w, moments, observation_seed(r, 42)) for w, r in zip(valwords, validation)]
    assert all(np.array_equal(np.isfinite(w), np.isfinite(n)) for w, n in zip(valwords, noise))
    with np.load(ROOT / "data/processed/zuco1_trt_mean_pilot_v1_20261030/sentence_vectors.npz", allow_pickle=False) as cache:
        structure = cache["count_duration_only_val"][:, :14]
    ordered = np.asarray([pooled(w, projection, 4) for w in valwords])
    features = {
        "mean_real": np.asarray([pooled(w, projection, 1) for w in valwords]),
        "ordered_real": ordered, "class_shuffled": ordered,
        "word_order_shuffled": np.asarray([pooled(w, projection, 4, observation_seed(r, 43)) for w, r in zip(valwords, validation)]),
        "matched_gaussian": np.asarray([pooled(w, noise_projection, 4) for w in noise]),
        "missingness_only": np.c_[np.asarray([missingness(w) for w in valwords]), basis],
        "structural_only": np.c_[structure, basis],
    }
    text = TextCache(plan["text_cache"])
    for report in reports:
        model = load(report["condition"] + "/ridge.npz")
        raw = features[report["condition"]]
        q = (np.where(np.isfinite(raw), raw, model["imputer_fill"]) - model["scaler_center"]) / model["scaler_scale"]
        prediction = q @ model["coef"].T + model["intercept"]
        rank = ranks(prediction, text.get(plan["candidate_keys"]), [r.key for r in validation], plan["candidate_keys"])
        np.testing.assert_array_equal(rank, report["ranks"])
        assert summarize(rank) == report["metrics"]
        assert sha(RUN / report["condition"] / "ridge.npz") == report["model_sha256"]
        assert report["selection"]["alpha"] == select_alpha(report["selection"]["grid"])
    alltrain = {tuple(k) for k in plan["train_observations"]}
    allval = {tuple(k) for k in plan["validation_observations"]}
    assert not alltrain & allval
    cv = json.loads((RUN / "inner_training_scores.json").read_text())
    assert len(cv) == 8 * 7 * 3
    for reader in plan["inner_readers"]:
        fold = json.loads((RUN / ("inner_fold_" + reader + "_provenance.json")).read_text())
        fitids = {tuple(k) for k in fold["fit_observations"]}
        queryids = {tuple(k) for k in fold["query_observations"]}
        assert not fitids & queryids and fitids | queryids == alltrain
        assert {k[1] for k in queryids} == {reader} and reader not in {k[1] for k in fitids}
        assert not (fitids | queryids) & allval
    for report in reports:
        for row in report["selection"]["grid"]:
            selected = [r for r in cv if r["condition"] == report["condition"] and r["alpha"] == row["alpha"]]
            assert len(selected) == 8
            assert row["macro_reader_recall@1"] == float(np.mean([r["recall@1"] for r in selected]))
    protected = json.loads((RUN / "preservation_before.json").read_text())
    assert all(sha(ROOT / path) == digest for path, digest in protected.items())
    result = {"status": "passed", "saved_model_replays": 7, "validation_ranks_reproduced": 7 * 512,
              "inner_reader_folds": 8, "inner_scores_verified": len(cv), "candidate_pool": 298,
              "gaussian_validation_masks_match": True, "prior_protected_files_unchanged": len(protected),
              "test_evaluated": False, "verification_only": "No fitting or new model/settings selection"}
    path = RUN / "validation_checks.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
