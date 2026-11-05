"""Frozen word-order pilot: inner training-reader selection, outer validation only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, sentence_key, observation_seed
from eeg_decoding.data.trt_view import iter_trt_sentences
from eeg_decoding.evaluation.controls import shuffled_keys
from eeg_decoding.evaluation.spectral_pilot import fit_word_moments, matched_gaussian, count_basis, fit_transform
from eeg_decoding.evaluation.incremental_ridge import ridge_path_predictions
from eeg_decoding.evaluation.word_order import fit_projector, pooled, missingness, select_alpha
from eeg_decoding.evaluation.retrieval import ranks, summarize, chance
from eeg_decoding.training.alignment_train import TextCache

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/word_order_pilot_v1.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity(record):
    return record.dataset, record.subject, record.session, record.task, record.sentence_id


def context():
    old = json.loads((ROOT / "results/spectral_trt_pilot_v1_20261030/plan.json").read_text())
    records = {identity(r): r for r in read_manifest(old["manifest"])}
    train = [records[tuple(k)] for k in old["train_observations"]]
    val = [records[tuple(k)] for k in old["validation_observations"]]
    assert len(train) == 2245 and len(val) == 512
    assert not {r.subject for r in train + val} & {"ZAB", "ZKB"}
    text = TextCache(old["text_cache"])
    assert text.digest == old["text_cache_sha256"] and sha(old["manifest"]) == old["manifest_sha256"]
    return old, train, val, text


def prepare(output):
    if output.exists():
        raise FileExistsError("Preserve existing artifacts; use a fresh output directory")
    cfg = json.loads(CFG.read_text())
    old, train, val, _ = context()
    protected = [p for p in (ROOT / "results").rglob("*") if p.is_file()]
    protected += [p for p in (ROOT / "docs").rglob("*") if p.is_file()]
    protected += list((ROOT / "data/processed/zuco1_trt_mean_pilot_v1_20261030").glob("*"))
    protected += [Path(old["manifest"]), Path(old["text_cache"])]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected if p.is_file()}
    output.mkdir(parents=True)
    write(output / "preservation_before.json", before)
    plan = {"configuration": cfg, "configuration_sha256": sha(CFG),
            "source_pilot_plan_sha256": sha(ROOT / "results/spectral_trt_pilot_v1_20261030/plan.json"),
            "train_observations": old["train_observations"], "validation_observations": old["validation_observations"],
            "candidate_keys": old["candidate_keys"], "candidate_pool_sha256": old["candidate_pool_sha256"],
            "inner_candidate_keys": sorted({r.key for r in train}),
            "inner_readers": sorted({r.subject for r in train}),
            "manifest": old["manifest"], "manifest_sha256": old["manifest_sha256"],
            "text_cache": old["text_cache"], "text_cache_sha256": old["text_cache_sha256"],
            "test_features_extracted": False, "test_evaluated": False, "new_exclusions": [],
            "word_sequence_meaning": "Source stimulus word order, not inferred fixation chronology or timing",
            "limitations": old["provenance_limitations"] + [
                "Word positions and missingness are gaze/stimulus-assisted; word identity is not a numeric input",
                "PCA and four word-position bins are a small linear sequence summary, not a deep sequence model",
                "Reused validation readers are not independent confirmation; no positive-result claim from one score"],
            "authorization": "User requested a new improvement pilot on 2026-10-31; prior stop remains in force for test evaluation"}
    write(output / "plan.json", plan)
    for source in [CFG, Path(__file__), ROOT / "src/eeg_decoding/evaluation/word_order.py",
                   ROOT / "src/eeg_decoding/data/trt_view.py", ROOT / "src/eeg_decoding/evaluation/spectral_pilot.py",
                   ROOT / "src/eeg_decoding/evaluation/incremental_ridge.py", ROOT / "tests/test_word_order.py"]:
        dest = output / "implementation" / source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
    wanted = defaultdict(dict)
    for r in train + val:
        wanted[r.subject][int(r.sentence_id)] = r
    previous = json.loads((ROOT / "data/processed/zuco1_trt_mean_pilot_v1_20261030/index.json").read_text())
    expected = {s["subject"]: s["sha256"] for s in previous["sources"]}
    data, provenance = {}, []
    for subject, observations in sorted(wanted.items()):
        source = ROOT / f"data/raw/zuco1/results{subject}_NR.mat"
        print(f"EXTRACT word TRT {subject}; train/validation only", flush=True)
        digest = sha(source)
        if digest != expected[subject]:
            raise ValueError("Source file changed")
        found = set()
        for slot, parent, words, _, _ in iter_trt_sentences(source):
            if slot not in observations:
                continue
            r = observations[slot]
            if parent != r.text or sentence_key(parent) != r.key:
                raise ValueError("Pairing mismatch")
            finite = np.isfinite(words)
            if words.shape[1] != 840 or not np.all(finite.all(1) | ~finite.any(1)):
                raise ValueError("Unreviewed word feature/missingness shape")
            data[identity(r)] = words
            found.add(slot)
        if found != set(observations):
            raise ValueError("Missing eligible observations")
        provenance.append({"subject": subject, "path": str(source.resolve()), "sha256": digest, "observations": len(found)})
        print(f"PAIRED {subject}: {len(found)}", flush=True)
    payload, coverage = {}, []
    for split, rows in [("train", train), ("val", val)]:
        sequences = [data[identity(r)] for r in rows]
        offsets = np.r_[0, np.cumsum([len(s) for s in sequences])]
        payload[split + "_words"] = np.concatenate(sequences)
        payload[split + "_offsets"] = offsets
        coverage.append({"split": split, "observations": len(rows), "words": int(offsets[-1]),
                         "complete_words": int(np.isfinite(payload[split + "_words"]).all(1).sum()),
                         "all_missing_observations": [list(identity(r)) for r, w in zip(rows, sequences) if not np.isfinite(w).any()]})
    np.savez_compressed(output / "word_sequences.npz", **payload)
    write(output / "feature_view.json", {"sources": provenance, "coverage": coverage,
          "cache_sha256": sha(output / "word_sequences.npz"), "test_features_extracted": False,
          "fields": "TRT_t1,t2,a1,a2,b1,b2,g1,g2; 105 channels per band; no word text stored"})
    print("PREPARATION COMPLETE", flush=True)


def sequence_features(train_words, query_words, train_records, query_records, raw_structure, cfg, model_dir=None):
    """Every fitted quantity uses the supplied training readers, never queries."""
    moments = fit_word_moments(train_words)
    real_projection = fit_projector(train_words, cfg["pca_components"], cfg["pca_max_fit_words"], cfg["seed"])
    gaussian_train = [matched_gaussian(w, moments, observation_seed(r, cfg["seed"])) for r, w in zip(train_records, train_words)]
    gaussian_query = [matched_gaussian(w, moments, observation_seed(r, cfg["seed"])) for r, w in zip(query_records, query_words)]
    gaussian_projection = fit_projector(gaussian_train, cfg["pca_components"], cfg["pca_max_fit_words"], cfg["seed"])
    result = {}
    for label, words, noise, rows in [("train", train_words, gaussian_train, train_records),
                                      ("query", query_words, gaussian_query, query_records)]:
        real = np.asarray([pooled(w, real_projection, cfg["segments"]) for w in words])
        counts = np.array([len(w) for w in words])
        basis, _ = count_basis(np.array([len(w) for w in train_words]), counts)
        result[label] = {
            "mean_real": np.asarray([pooled(w, real_projection, 1) for w in words]),
            "ordered_real": real,
            "matched_gaussian": np.asarray([pooled(w, gaussian_projection, cfg["segments"]) for w in noise]),
            "word_order_shuffled": np.asarray([pooled(w, real_projection, cfg["segments"], observation_seed(r, cfg["seed"] + 1)) for r, w in zip(rows, words)]),
            "class_shuffled": real,
            "missingness_only": np.c_[np.asarray([missingness(w, cfg["segments"]) for w in words]), basis],
            "structural_only": np.c_[raw_structure[label][:, :14], basis],
        }
        assert all(len(a) == len(rows) and np.isfinite(a).any() for a in result[label].values())
    if model_dir:
        for name, projection in [("real", real_projection), ("gaussian", gaussian_projection)]:
            np.savez_compressed(model_dir / (name + "_word_projection.npz"), **projection)
        np.savez_compressed(model_dir / "gaussian_training_moments.npz", **moments)
        write(model_dir / "word_projection_provenance.json", {
            "fit_observations": [list(identity(r)) for r in train_records],
            "fit_words": real_projection["training_words"], "sampled_words": real_projection["pca_fit_words"],
            "components": cfg["pca_components"],
            "real_explained_variance_ratio_sum": float(real_projection["explained_variance_ratio"].sum()),
            "gaussian_masks_match": all(np.array_equal(np.isfinite(a), np.isfinite(b)) for a, b in zip(train_words + query_words, gaussian_train + gaussian_query))})
    return result


def fit(output):
    if (output / "validation_summary.json").exists():
        raise FileExistsError("Already evaluated; preserve results")
    plan = json.loads((output / "plan.json").read_text())
    cfg = plan["configuration"]
    if sha(CFG) != plan["configuration_sha256"]:
        raise ValueError("Frozen configuration changed")
    for name in ("run_word_order_pilot.py", "word_order.py"):
        source = Path(__file__) if name.startswith("run_") else ROOT / "src/eeg_decoding/evaluation" / name
        if sha(source) != sha(output / "implementation" / name):
            raise ValueError("Frozen implementation changed")
    view = json.loads((output / "feature_view.json").read_text())
    if sha(output / "word_sequences.npz") != view["cache_sha256"]:
        raise ValueError("Cache changed")
    old, train, val, text = context()
    with np.load(output / "word_sequences.npz", allow_pickle=False) as cache:
        assert set(cache.files) == {"train_words", "train_offsets", "val_words", "val_offsets"}
        sequences = {}
        for split in ("train", "val"):
            x, offsets = cache[split + "_words"], cache[split + "_offsets"]
            sequences[split] = [x[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]
    with np.load(ROOT / "data/processed/zuco1_trt_mean_pilot_v1_20261030/sentence_vectors.npz", allow_pickle=False) as cache:
        structure = {s: cache["count_duration_only_" + s][:, :14].copy() for s in ("train", "val")}
    table = []
    for reader in plan["inner_readers"]:
        started = time.monotonic()
        a = np.array([i for i, r in enumerate(train) if r.subject != reader])
        b = np.array([i for i, r in enumerate(train) if r.subject == reader])
        fit_rows, query_rows = [train[i] for i in a], [train[i] for i in b]
        print(f"INNER training-reader holdout {reader}: fit={len(a)} query={len(b)}", flush=True)
        matrices = sequence_features([sequences["train"][i] for i in a], [sequences["train"][i] for i in b],
            fit_rows, query_rows, {"train": structure["train"][a], "query": structure["train"][b]}, cfg)
        pool = plan["inner_candidate_keys"]
        write(output / ("inner_fold_" + reader + "_provenance.json"), {
            "fit_observations": [list(identity(r)) for r in fit_rows],
            "query_observations": [list(identity(r)) for r in query_rows], "candidate_keys": pool,
            "query_classes_absent_from_fit": sorted({r.key for r in query_rows} - {r.key for r in fit_rows}),
            "preprocessing_fit_scope": "Only fit_observations, including Gaussian moments and PCA sampling"})
        for condition in cfg["conditions"]:
            x, q, _ = fit_transform(matrices["train"][condition], matrices["query"][condition], "median", "standard")
            keys = [r.key for r in fit_rows]
            if condition == "class_shuffled":
                keys = shuffled_keys(keys, cfg["seed"])
            for alpha, prediction, _ in ridge_path_predictions(x, q, text.get(keys).astype(float), cfg["ridge_alphas"]):
                metric = summarize(ranks(prediction, text.get(pool), [r.key for r in query_rows], pool))
                table.append({"reader": reader, "condition": condition, "alpha": alpha, **metric})
        write(output / "inner_training_scores.json", table)
        print(f"INNER COMPLETE {reader}: {time.monotonic() - started:.1f}s", flush=True)
    selection = {}
    for condition in cfg["conditions"]:
        grid = []
        for alpha in cfg["ridge_alphas"]:
            rows = [r for r in table if r["condition"] == condition and r["alpha"] == alpha]
            assert len(rows) == len(plan["inner_readers"])
            grid.append({"alpha": alpha, "macro_reader_recall@1": float(np.mean([r["recall@1"] for r in rows])),
                         "macro_reader_mrr": float(np.mean([r["mrr"] for r in rows]))})
        selection[condition] = {"alpha": select_alpha(grid), "grid": grid,
                                "selection_scope": "Eight original training readers only; Recall@1"}
    write(output / "frozen_selection.json", selection)
    print("SETTINGS FROZEN before any outer validation metrics", flush=True)
    matrices = sequence_features(sequences["train"], sequences["val"], train, val,
                                 {"train": structure["train"], "query": structure["val"]}, cfg, output)
    reports = []
    for condition in cfg["conditions"]:
        folder = output / condition
        folder.mkdir(exist_ok=False)
        x, q, transform = fit_transform(matrices["train"][condition], matrices["query"][condition], "median", "standard")
        keys = [r.key for r in train]
        if condition == "class_shuffled":
            shuffled = shuffled_keys(keys, cfg["seed"])
            write(folder / "class_mapping.json", dict(sorted(set(zip(keys, shuffled)))))
            keys = shuffled
        alpha, prediction, fitted = next(ridge_path_predictions(x, q, text.get(keys).astype(float), [selection[condition]["alpha"]]))
        np.savez_compressed(folder / "ridge.npz", **fitted, imputer_fill=transform["fill"],
                            scaler_center=transform["center"], scaler_scale=transform["scale"])
        pool = plan["candidate_keys"]
        rank = ranks(prediction, text.get(pool), [r.key for r in val], pool)
        report = {"condition": condition, "evaluation_split": "val", "metrics": summarize(rank),
                  "chance": chance(len(pool)), "candidate_keys": pool, "candidate_pool_sha256": plan["candidate_pool_sha256"],
                  "observation_identities": plan["validation_observations"], "ranks": rank.tolist(),
                  "per_subject": {s: summarize(rank[[r.subject == s for r in val]]) for s in sorted({r.subject for r in val})},
                  "selection": selection[condition], "n_features": x.shape[1],
                  "ridge_parameter_count": int(fitted["coef"].size + fitted["intercept"].size),
                  "model_sha256": sha(folder / "ridge.npz")}
        write(folder / "validation.json", report)
        reports.append(report)
        print(json.dumps({"condition": condition, "alpha": alpha, "metrics": report["metrics"]}), flush=True)
    write(output / "validation_summary.json", reports)
    by = {r["condition"]: r for r in reports}
    real = by["ordered_real"]
    gaps = [{"comparison": "ordered_real_minus_" + c,
             **{m: real["metrics"][m] - by[c]["metrics"][m] for m in ("recall@1", "recall@5", "recall@10", "mrr")}}
            for c in cfg["conditions"] if c != "ordered_real"]
    checks = {c + "_aggregate": all(real["metrics"][m] > by[c]["metrics"][m]
                                    for m in ("recall@1", "mrr")) for c in cfg["conditions"] if c != "ordered_real"}
    for s in real["per_subject"]:
        for c in ("matched_gaussian", "structural_only"):
            checks[s + "_beats_" + c] = real["per_subject"][s]["recall@1"] > by[c]["per_subject"][s]["recall@1"]
    write(output / "real_minus_controls.json", gaps)
    write(output / "decision.json", {"descriptive_screen_passed": all(checks.values()), "checks": checks,
          "test_evaluated": False, "next_action": "Stop; no tuning or test evaluation; report exploratory validation only"})
    before = json.loads((output / "preservation_before.json").read_text())
    changed = [p for p, digest in before.items() if sha(ROOT / p) != digest]
    if changed:
        raise RuntimeError(f"Protected artifacts changed: {changed}")
    write(output / "preservation_after.json", {"unchanged": True, "protected_files": len(before), "test_evaluated": False})
    print("COMPLETE; preserved prior files; test untouched", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-prepared", action="store_true", help="Fit a previously prepared frozen run; refuses completed runs")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if not args.fit_prepared:
            prepare(args.output.resolve())
        fit(args.output.resolve())
