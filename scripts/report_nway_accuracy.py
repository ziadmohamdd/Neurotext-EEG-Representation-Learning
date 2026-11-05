"""Replay latest frozen sentence models; compute exact expected N-way accuracy."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode
from eeg_decoding.evaluation.spectral_pilot import matched_gaussian, count_basis
from eeg_decoding.evaluation.retrieval import normalize
from eeg_decoding.evaluation.nway import expected_top1, score_counts
from run_gru_retrieval_loss import ROOT, SOURCE, sha, write, ident, predict

RUN = ROOT / "results/word_pretrain_v1_20261101"
OUTPUT = ROOT / "results/nway_word_pretrain_v1_20261101"
NS = (2, 5, 10, 20, 50, 100, 300)


def read(path):
    return json.loads(path.read_text())


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve prior N-way analysis")
    plan = read(RUN / "plan.json")
    cfg = plan["configuration"]
    pool = plan["training_candidate_keys"]
    assert len(pool) == len(set(pool)) == 300
    records = {tuple(ident(r)): r for r in read_manifest(plan["manifest"])}
    rows = [records[tuple(k)] for k in plan["train_observations"]]
    assert len(rows) == 2245 and not {r.subject for r in rows} & {"ZJS", "ZKW", "ZAB", "ZKB"}
    by_id = {tuple(ident(r)): i for i, r in enumerate(rows)}
    text = TextCache(plan["text_cache"])
    with np.load(SOURCE / "word_sequences.npz", allow_pickle=False) as z:
        words, offsets = z["train_words"], z["train_offsets"]
    seq = [words[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    with np.load(ROOT / "data/processed/zuco1_trt_mean_pilot_v1_20261030/sentence_vectors.npz", allow_pickle=False) as z:
        structural = z["count_duration_only_train"][:, :14]
    protected_files = [p for d in ("results", "docs", "checkpoints") for p in (ROOT / d).rglob("*") if p.is_file()]
    protected_files += [Path(plan["manifest"]), Path(plan["text_cache"])]
    protected = {str(p.relative_to(ROOT)): sha(p) for p in protected_files}
    OUTPUT.mkdir(parents=True)
    write(OUTPUT / "preservation_before.json", protected)
    write(OUTPUT / "plan.json", {"source_run": str(RUN.relative_to(ROOT)), "source_plan_sha256": sha(RUN / "plan.json"),
          "N": list(NS), "pool_size": 300, "candidate_keys": pool,
          "train_observations": plan["train_observations"], "aggregation": "Equal-reader macro mean across original eight training-CV readers",
          "calculation": "Exact expected strict top-1 under uniform N-1 distractors without replacement from original 299 incorrect candidates",
          "tie_policy": "Truth must strictly outrank every distractor; matches original average-tied-rank <= 1",
          "sampling": "Analytical expectation over all possible pools; no Monte Carlo draws. Every condition uses identical candidate universe and uniform draw mechanism.",
          "no_training_or_selection": True, "outer_validation_or_test_evaluated": False,
          "limitation": "Uniform random distractors only; not semantic-hard or word-count-matched pools. Smaller N changes task difficulty, not model quality."})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocabulary = {key: i for i, key in enumerate(pool)}
    candidates = normalize(text.get(pool))
    statistics, per_reader = [], []
    cache = {}
    for folder in sorted(RUN.glob("cv_*")):
        provenance = read(folder / "provenance.json")
        a = [by_id[tuple(k)] for k in provenance["fit_observations"]]
        b = [by_id[tuple(k)] for k in provenance["query_observations"]]
        assert set(a).isdisjoint(b) and set(a) | set(b) == set(range(len(rows)))
        query = [rows[i] for i in b]
        targets = np.array([vocabulary[r.key] for r in query])
        with np.load(folder / "preprocessing.npz", allow_pickle=False) as z:
            pp = {key: z[key] for key in z.files}
        for arm in cfg["arms"] + ["structural_only"]:
            if arm == "structural_only":
                counts_a = np.array([len(seq[i]) for i in a])
                counts_b = np.array([len(seq[i]) for i in b])
                basis, _ = count_basis(counts_a, counts_b)
                with np.load(folder / "structural_ridge.npz", allow_pickle=False) as z:
                    raw = np.c_[structural[b], basis]
                    design = (np.where(np.isfinite(raw), raw, z["fill"]) - z["center"]) / z["scale"]
                    prediction = design @ z["coef"].T + z["intercept"]
                saved = next(r for r in read(folder / "sentence_scores.json") if r["arm"] == arm)
            else:
                query_seq = [seq[i] for i in b]
                scaler = {key: pp["real_" + key] for key in ("center", "scale")}
                if arm == "pretrained_gaussian":
                    moments = {key: pp["moment_" + key] for key in ("mean", "sd")}
                    query_seq = [matched_gaussian(w, moments, observation_seed(r, cfg["seed"])) for w, r in zip(query_seq, query)]
                    scaler = {key: pp["gaussian_" + key] for key in ("center", "scale")}
                    assert all(np.array_equal(np.isfinite(w), np.isfinite(seq[i])) for w, i in zip(query_seq, b))
                condition = "missingness_only" if arm == "pretrained_missingness" else "real_eeg"
                xs = [encode(w, scaler, condition) for w in query_seq]
                model = SequenceGRU(384, cfg["projection"], cfg["hidden"]).to(device)
                checkpoint = torch.load(folder / arm / "sentence_final.pt", map_location=device, weights_only=True)
                model.load_state_dict(checkpoint["model"])
                prediction = predict(model, xs, device, cfg["sentence_batch_size"])
                saved = read(folder / arm / "sentence_holdout.json")
            unit = normalize(prediction)
            below, ties, rank = [], [], []
            # Match the original rank function's matrix multiplication chunking exactly.
            for start in range(0, len(unit), 128):
                bb, tt, rr = score_counts(unit[start:start + 128] @ candidates.T, targets[start:start + 128])
                below.extend(bb.tolist()); ties.extend(tt.tolist()); rank.extend(rr.tolist())
            np.testing.assert_array_equal(rank, saved["ranks"])
            cache[folder.name + "_" + arm + "_strictly_below"] = np.asarray(below, dtype=np.int16)
            cache[folder.name + "_" + arm + "_ties"] = np.asarray(ties, dtype=np.int16)
            statistics.append({"reader": folder.name[3:], "arm": arm, "n_queries": len(b),
                               "target_tie_queries": int(np.sum(np.asarray(ties) > 1)), "rank_replay_matches": True,
                               "observation_identities": provenance["query_observations"]})
            for n in NS:
                probabilities = expected_top1(below, len(pool), n)
                if n == len(pool):
                    assert probabilities.mean() == np.mean(np.asarray(rank) <= 1)
                per_reader.append({"reader": folder.name[3:], "arm": arm, "N": n,
                                   "expected_accuracy": float(probabilities.mean()), "chance": 1 / n, "n_queries": len(b)})
        print(folder.name, "replayed six frozen neural models and structural model", flush=True)
    summary = []
    old_summary = read(RUN / "training_cv_summary.json")
    for n in NS:
        for arm in cfg["arms"] + ["structural_only"]:
            matches = [r for r in per_reader if r["N"] == n and r["arm"] == arm]
            assert len(matches) == 8
            accuracy = float(np.mean([r["expected_accuracy"] for r in matches]))
            if n == len(pool):
                assert abs(accuracy - old_summary[arm]["macro"]["recall@1"]) < 1e-12
            summary.append({"N": n, "arm": arm, "expected_accuracy": accuracy,
                            "chance": 1 / n, "minus_chance_pp": 100 * (accuracy - 1 / n)})
    np.savez_compressed(OUTPUT / "strict_comparison_counts.npz", **cache)
    write(OUTPUT / "per_reader.json", per_reader)
    write(OUTPUT / "summary.json", summary)
    write(OUTPUT / "replay_details.json", statistics)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    assert all(sha(ROOT / path) == digest for path, digest in protected.items())
    write(OUTPUT / "verification.json", {"passed": True, "neural_model_replays": 48, "structural_replays": 8,
          "original_300way_ranks_reproduced": 2245 * 7, "300way_macro_scores_match_all_original_arms": True,
          "prior_protected_files_unchanged": len(protected), "no_training": True,
          "outer_validation_or_test_evaluated": False})
    archive = ROOT / "docs/research_results/nway_word_pretrain_v1_20261101"
    if archive.exists():
        raise FileExistsError("Preserve prior report")
    archive.mkdir(parents=True)
    lines = ["# Expected N-way retrieval accuracy — latest word-pretrained GRU", "",
        "These are exact expectations under uniformly sampled distractors, computed from replayed "
        "frozen models on the original eight training-reader CV folds. They are not new independent "
        "validation/test results, Monte Carlo estimates or improvements to the trained model.", "",
        "Each pool contains the correct sentence and N−1 unique distractors drawn without replacement "
        "from the original 299 incorrect sentences. For b strictly lower-scoring candidates out of M=300, "
        "P(strict top-1)=C(b,N−1)/C(M−1,N−1). Ties with the truth count as a miss, matching the original "
        "average-tied-rank<=1 criterion. All arms use the same candidate universe and uniform sampling "
        "mechanism; exact expectation removes sampling variability. The probabilities are averaged "
        "within reader then equally across the eight readers (2,245 total observations).", "",
        "Random chance is 1/N. Uniform random distractors do not control sentence length or semantic "
        "similarity. Hard-negative or length-matched pools would be a different evaluation. Repeated "
        "candidate subsets are not new independent participants or cohorts. No setting was selected "
        "from these N-way scores, and no raw source, outer-validation or test EEG array was used.", "",
        "## Expected top-1 accuracy (%)", "",
        "| N | Pretrained real EEG | Gaussian | Class-shuffled | Missingness-only | Structural-only | Chance |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for n in NS:
        by = {r["arm"]: r["expected_accuracy"] for r in summary if r["N"] == n}
        cells = [n, *[f"{100*by[c]:.4f}" for c in ("pretrained_real", "pretrained_gaussian", "pretrained_class_shuffled", "pretrained_missingness", "structural_only")], f"{100/n:.4f}"]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    lines += ["", "Fresh equal-update and same-sentence-update baseline results, all reader-level "
              "values and exact score comparison counts are also archived. Smaller N raises chance "
              "and changes difficulty; higher raw percentages must not be presented as improved EEG decoding.", "",
              f"Three focused analytical tests passed before execution. Replays reproduced all "
              f"15,715 original ranks and every original 300-way macro Recall@1. All {len(protected)} "
              "protected prior files remained SHA256-identical. No fitting, tuning or new checkpoint "
              "selection occurred. The .venv and prior reports/results are unchanged."]
    (archive / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = []
    for source in list(OUTPUT.glob("*.json")) + [OUTPUT / "summary.csv", OUTPUT / "strict_comparison_counts.npz", Path(__file__),
                  ROOT / "src/eeg_decoding/evaluation/nway.py", ROOT / "tests/test_nway.py"]:
        rel = source.relative_to(OUTPUT) if OUTPUT in source.parents else Path("implementation") / source.name
        dest = archive / rel
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
        manifest.append({"source": str(source.relative_to(ROOT)), "archive": str(rel), "sha256": sha(source)})
    write(archive / "archive_manifest.json", manifest)
    for n in NS:
        subset = {r["arm"]: r for r in summary if r["N"] == n}
        print(f"{n:3d}-way: real={100*subset['pretrained_real']['expected_accuracy']:.4f}% "
              f"Gaussian={100*subset['pretrained_gaussian']['expected_accuracy']:.4f}% "
              f"mask={100*subset['pretrained_missingness']['expected_accuracy']:.4f}% "
              f"structure={100*subset['structural_only']['expected_accuracy']:.4f}% chance={100/n:.4f}%")
    print(archive / "README.md")


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        main()
