"""Bounded training-reader diagnosis; never extract outer validation/test arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.data.trt_view import mean_features
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments, matched_gaussian, count_basis, fit_transform
from eeg_decoding.evaluation.incremental_ridge import ridge_path_predictions
from eeg_decoding.evaluation.relative_trt import relative_mean, score_ranks, choose
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.retrieval import summarize, ranks
from eeg_decoding.training.alignment_train import TextCache

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/amplitude_audit_v1.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(r):
    return r.dataset, r.subject, r.session, r.task, r.sentence_id


def main(output):
    if output.exists():
        raise FileExistsError("Preserve prior results; choose a new versioned directory")
    cfg = json.loads(CONFIG.read_text())
    source = ROOT / "results/word_order_pilot_v1_20261031"
    old = json.loads((source / "plan.json").read_text())
    source_view = json.loads((source / "feature_view.json").read_text())
    if sha(source / "word_sequences.npz") != source_view["cache_sha256"]:
        raise ValueError("Original word cache changed")
    protected = [p for d in ("results", "docs", "checkpoints") for p in (ROOT / d).rglob("*") if p.is_file()]
    protected += [Path(old["manifest"]), Path(old["text_cache"])]
    protected += [p for p in (ROOT / "data/processed/zuco1_trt_mean_pilot_v1_20261030").glob("*") if p.is_file()]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    output.mkdir(parents=True)
    write(output / "preservation_before.json", before)
    plan = {"configuration": cfg, "config_sha256": sha(CONFIG), "source_word_cache_sha256": sha(source / "word_sequences.npz"),
            "train_observations": old["train_observations"], "candidate_keys": old["inner_candidate_keys"],
            "readers": old["inner_readers"], "outer_validation_extracted": False, "test_evaluated": False,
            "word_sequence_arrays_read": ["train_words", "train_offsets"], "structural_arrays_read": ["count_duration_only_train"],
            "prior_pilot_pca_variance_retained": .7524881127874974,
            "purpose": "One bounded training-only diagnostic. No follow-up based on outer validation or test scores.",
            "candidate_pool_sha256": hashlib.sha256(json.dumps(old["inner_candidate_keys"]).encode()).hexdigest(),
            "source_limitations": old["limitations"],
            "authorization": "User requested further investigation and a justified improvement attempt on 2026-11-01"}
    write(output / "plan.json", plan)
    for p in [CONFIG, Path(__file__), ROOT / "src/eeg_decoding/evaluation/relative_trt.py", ROOT / "tests/test_relative_trt.py",
              ROOT / "src/eeg_decoding/evaluation/incremental_ridge.py", ROOT / "src/eeg_decoding/evaluation/spectral_pilot.py"]:
        dest = output / "implementation" / p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, dest)
    with np.load(source / "word_sequences.npz", allow_pickle=False) as z:
        words, offsets = z["train_words"], z["train_offsets"]
    sequences = [words[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    rows = {identity(r): r for r in read_manifest(old["manifest"])}
    train = [rows[tuple(k)] for k in plan["train_observations"]]
    assert len(train) == len(sequences) == 2245 and len(plan["candidate_keys"]) == 300
    assert not {r.subject for r in train} & {"ZJS", "ZKW", "ZAB", "ZKB"}
    text = TextCache(old["text_cache"])
    assert text.digest == old["text_cache_sha256"]
    pool = plan["candidate_keys"]
    lookup = {k: i for i, k in enumerate(pool)}
    keys = [r.key for r in train]
    labels = np.array([lookup[k] for k in keys])
    targets = np.c_[text.get(keys), np.eye(len(pool))[labels]]
    width = text.get(pool).shape[1]
    with np.load(ROOT / "data/processed/zuco1_trt_mean_pilot_v1_20261030/sentence_vectors.npz", allow_pickle=False) as z:
        structural = z["count_duration_only_train"][:, :14]
    real = np.asarray([relative_mean(w) for w in sequences])
    raw = np.asarray([mean_features(w) for w in sequences])
    finite_words = words[np.isfinite(words).all(1)]
    audit = {"train_word_value_percentiles": np.percentile(finite_words, [0, 1, 25, 50, 75, 99, 100]).tolist(),
             "negative_real_values": int((finite_words < 0).sum()), "complete_training_words": len(finite_words),
             "text_candidate_common_mean_norm": float(np.linalg.norm(text.get(pool).mean(0))),
             "reader_raw_mean_norms": {}, "reader_relative_mean_norms": {},
             "all_missing_training_observations": [list(identity(r)) for r, x in zip(train, real) if not np.isfinite(x).any()],
             "implementation_findings": "No confirmed pairing/ranking bug. PCA loss and amplitude differences are hypotheses, not proven causes."}
    subjects = np.array([r.subject for r in train])
    for reader in plan["readers"]:
        audit["reader_raw_mean_norms"][reader] = float(np.linalg.norm(np.nanmean(raw[subjects == reader], 0)))
        audit["reader_relative_mean_norms"][reader] = float(np.linalg.norm(np.nanmean(real[subjects == reader], 0)))
    write(output / "training_feature_audit.json", audit)
    scores = []
    for reader in plan["readers"]:
        start = time.monotonic()
        a, b = np.flatnonzero(subjects != reader), np.flatnonzero(subjects == reader)
        print(f"TRAINING-ONLY HOLDOUT {reader} fit={len(a)} query={len(b)}", flush=True)
        fitwords = [sequences[i] for i in a]
        moments = fit_word_moments(fitwords)
        noise = np.asarray([relative_mean(matched_gaussian(w, moments, observation_seed(r, cfg["seed"])))
                            for r, w in zip(train, sequences)])
        # All draws share the real mask; only remaining-reader words fit moments.
        np.testing.assert_array_equal(np.isnan(noise), np.isnan(real))
        counts = np.array([len(w) for w in sequences])
        sbasis_a, _ = count_basis(counts[a], counts[a])
        sbasis_b, _ = count_basis(counts[a], counts[b])
        sa, sb, _ = fit_transform(np.c_[structural[a], sbasis_a], np.c_[structural[b], sbasis_b], "median", "standard")
        ra, rb, _ = fit_transform(real[a], real[b], "median", "standard")
        ga, gb, _ = fit_transform(noise[a], noise[b], "median", "standard")
        shuffle = shuffled_keys([keys[i] for i in a], cfg["seed"])
        shuffled_targets = np.c_[text.get(shuffle), np.eye(len(pool))[[lookup[k] for k in shuffle]]]
        y = np.c_[targets[a], shuffled_targets]
        write(output / ("fold_" + reader + "_provenance.json"), {
            "fit_observations": [list(identity(train[i])) for i in a],
            "query_observations": [list(identity(train[i])) for i in b],
            "fit_readers": sorted(set(subjects[a])), "query_reader": reader,
            "train_classes": len(set(keys[i] for i in a)), "candidate_keys": pool,
            "class_shuffle_mapping": dict(sorted(set(zip([keys[i] for i in a], shuffle)))),
            "gaussian_masks_match": True, "all_preprocessing_fit_scope": "fit_observations only"})
        jobs = [("structural_only", 0., sa, sb), ("eeg_only", 1., ra, rb), ("gaussian_only", 1., ga, gb)]
        for weight in cfg["eeg_weights"]:
            jobs += [("structural_real", weight, np.c_[sa, weight * ra], np.c_[sb, weight * rb]),
                     ("structural_gaussian", weight, np.c_[sa, weight * ga], np.c_[sb, weight * gb])]
        # Zero weight paths directly reuse structural-only predictions, avoiding redundant SVDs.
        base = None
        for condition, weight, x, q in jobs:
            if condition.startswith("structural_") and condition != "structural_only" and weight == 0:
                path = base
            else:
                path = list(ridge_path_predictions(x, q, y, cfg["ridge_alphas"]))
                if condition == "structural_only":
                    base = path
            for alpha, prediction, _ in path:
                variants = [(condition, prediction[:, :targets.shape[1]])]
                if condition == "structural_real":
                    variants.append(("structural_shuffled", prediction[:, targets.shape[1]:]))
                for name, pred in variants:
                    semantic = ranks(pred[:, :width], text.get(pool), [keys[i] for i in b], pool)
                    categorical = score_ranks(pred[:, width:], labels[b])
                    for head, rank in [("semantic_embedding", semantic), ("known_sentence_class", categorical)]:
                        scores.append({"reader": reader, "head": head, "condition": name,
                                       "weight": weight, "alpha": alpha, **summarize(rank), "ranks": rank.tolist()})
        write(output / "inner_training_scores.json", scores)
        print(f"COMPLETE {reader}: {time.monotonic() - start:.1f}s", flush=True)
    summary = []
    for head in cfg["heads"]:
        for condition in cfg["conditions"]:
            ws = cfg["eeg_weights"] if condition in ("structural_real", "structural_gaussian", "structural_shuffled") else [0. if condition == "structural_only" else 1.]
            grid = []
            for weight in ws:
                for alpha in cfg["ridge_alphas"]:
                    matching = [r for r in scores if (r["head"], r["condition"], r["weight"], r["alpha"]) == (head, condition, weight, alpha)]
                    assert len(matching) == 8
                    grid.append({"weight": weight, "alpha": alpha,
                                 **{"macro_" + metric: float(np.mean([r[metric] for r in matching]))
                                    for metric in ("recall@1", "recall@5", "recall@10", "mrr")}})
            chosen = choose(grid)
            selected_rows = [r for r in scores if (r["head"], r["condition"], r["weight"], r["alpha"]) == (head, condition, chosen["weight"], chosen["alpha"])]
            result = {"head": head, "condition": condition, "selection": chosen, "grid": grid,
                      "per_reader": {r["reader"]: {k: r[k] for k in ("recall@1", "recall@5", "recall@10", "mrr", "n_queries")} for r in selected_rows}}
            summary.append(result)
            print(json.dumps({"head": head, "condition": condition, **chosen}), flush=True)
    write(output / "training_summary.json", summary)
    gates = []
    for head in cfg["heads"]:
        by = {r["condition"]: r for r in summary if r["head"] == head}
        realrow = by["structural_real"]
        gaps = {c: realrow["selection"]["macro_recall@1"] - by[c]["selection"]["macro_recall@1"] for c in ("structural_only", "structural_gaussian")}
        positive = {c: sum(realrow["per_reader"][s]["recall@1"] > by[c]["per_reader"][s]["recall@1"] for s in plan["readers"]) for c in gaps}
        gates.append({"head": head, "macro_recall_gaps": gaps, "positive_readers": positive,
                      "passed": all(v >= .02 for v in gaps.values()) and all(v >= 6 for v in positive.values())})
    write(output / "decision.json", {"gates": gates, "outer_validation_extracted": False, "test_evaluated": False,
                                    "next_action": "Stop training-only diagnostic; no additional search or outer evaluation"})
    changed = [p for p, digest in before.items() if sha(ROOT / p) != digest]
    if changed:
        raise RuntimeError(f"Protected artifacts changed: {changed}")
    write(output / "preservation_after.json", {"unchanged": True, "protected_files": len(before)})
    print("COMPLETE; outer validation and test untouched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        main(args.output.resolve())
