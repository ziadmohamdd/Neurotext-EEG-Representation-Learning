"""Verify/replay and archive a completed loss-only GRU run; never fit a model."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode
from eeg_decoding.evaluation.spectral_pilot import matched_gaussian
from eeg_decoding.evaluation.retrieval import normalize, ranks, summarize
from run_gru_pilot import predict, sha, ident

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/gru_retrieval_loss_v1_20261101"
BASE = ROOT / "results/gru_pilot_v1_20261101"
AUDIT = ROOT / "results/gru_prediction_audit_v1_20261101"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    if (RUN / "final_verification.json").exists():
        raise FileExistsError("Preserve final verification")
    plan = read(RUN / "plan.json")
    cfg = plan["configuration"]
    rows = read(RUN / "training_cv_all_scores.json")
    summary = read(RUN / "training_cv_summary.json")
    epoch = read(RUN / "frozen_selection.json")["common_epoch"]
    base_epoch = read(BASE / "frozen_selection.json")["common_epoch"]
    expected = {tuple(k) for k in plan["train_observations"]}
    pool = plan["training_candidate_keys"]
    assert len(pool) == len(set(pool)) == 300 and len(expected) == 2245 and len(rows) == 152
    old = read(ROOT / "results/word_order_pilot_v1_20261031/plan.json")
    records = {tuple(ident(r)): r for r in read_manifest(old["manifest"])}
    train = [records[tuple(k)] for k in plan["train_observations"]]
    assert not {r.subject for r in train} & {"ZJS", "ZKW", "ZAB", "ZKB"}
    lookup = {tuple(ident(r)): i for i, r in enumerate(train)}
    text = TextCache(old["text_cache"])
    assert text.digest == old["text_cache_sha256"]
    with np.load(ROOT / "results/word_order_pilot_v1_20261031/word_sequences.npz", allow_pickle=False) as z:
        w, offsets = z["train_words"], z["train_offsets"]
    sequences = [w[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    union, prediction_stats = set(), []
    for folder in sorted(RUN.glob("cv_*")):
        original = BASE / folder.name
        prov = read(folder / "provenance.json")
        a, b = {tuple(k) for k in prov["fit_observations"]}, {tuple(k) for k in prov["query_observations"]}
        assert a | b == expected and not a & b and not union & b
        union |= b
        assert prov == read(original / "provenance.json")
        current_weights = torch.load(folder / "initial_weights.pt", map_location="cpu", weights_only=True)
        old_weights = torch.load(original / "initial_weights.pt", map_location="cpu", weights_only=True)
        for key in current_weights:
            torch.testing.assert_close(current_weights[key], old_weights[key], rtol=0, atol=0)
        assert sum(v.numel() for v in current_weights.values()) == 45952
        with np.load(folder / "preprocessing.npz", allow_pickle=False) as z, np.load(original / "preprocessing.npz", allow_pickle=False) as oldz:
            assert set(z.files) == set(oldz.files)
            for key in z.files:
                np.testing.assert_array_equal(z[key], oldz[key])
            scalers = {name: {k: z[name + "_" + k] for k in ("center", "scale")} for name in ("real", "gaussian")}
            moments = {k: z["moments_" + k] for k in ("mean", "sd")}
        scores = read(folder / "scores.json")
        assert len(scores) == 19
        for score in scores:
            metric = summarize(score["ranks"])
            assert len(score["ranks"]) == len(b)
            for key in metric:
                assert abs(score[key] - metric[key]) < 1e-12
        structural = next(r for r in scores if r["condition"] == "structural_only")
        old_structural = next(r for r in read(original / "scores.json") if r["condition"] == "structural_only")
        assert structural == old_structural
        query = [records[tuple(k)] for k in prov["query_observations"]]
        query_words = [sequences[lookup[tuple(ident(r))]] for r in query]
        for condition in cfg["conditions"]:
            history = read(folder / condition / "history.json")
            assert [r["epoch"] for r in history] == list(range(1, 13))
            assert all(np.isfinite(r["training_candidate_ce"]) for r in history)
            for e in cfg["epoch_candidates"]:
                assert (folder / condition / f"epoch_{e}.pt").is_file()
            words = query_words
            scaler = scalers["real"]
            if condition == "matched_gaussian":
                words = [matched_gaussian(w, moments, observation_seed(r, cfg["seed"])) for w, r in zip(words, query)]
                scaler = scalers["gaussian"]
                assert all(np.array_equal(np.isfinite(a), np.isfinite(b)) for a, b in zip(words, query_words))
            xs = [encode(w, scaler, condition, observation_seed(r, cfg["seed"] + 1)) for w, r in zip(words, query)]
            model = SequenceGRU(384, cfg["projection"], cfg["hidden"]).to(device)
            model.load_state_dict(torch.load(folder / condition / f"epoch_{epoch}.pt", map_location=device, weights_only=True)["model"])
            pred = predict(model, xs, device, cfg["batch_size"])
            rank = ranks(pred, text.get(pool), [r.key for r in query], pool)
            saved = next(r for r in scores if r["condition"] == condition and r["epoch"] == epoch)
            np.testing.assert_array_equal(rank, saved["ranks"])
            winners = (normalize(pred) @ normalize(text.get(pool)).T).argmax(1)
            counts = np.array(list(Counter(winners.tolist()).values())) / len(winners)
            prediction_stats.append({"reader": folder.name[3:], "condition": condition,
                                     "unique_top1_sentences": len(counts), "largest_top1_share": float(counts.max()),
                                     "effective_top1_sentences": float(np.exp(-np.sum(counts * np.log(counts))))})
        print(folder.name, "initialization/preprocessing matched; six saved models replayed", flush=True)
    assert union == expected
    for condition, info in summary.items():
        matches = [r for r in rows if r["condition"] == condition and r["epoch"] == (0 if condition == "structural_only" else epoch)]
        assert len(matches) == 8
        for metric in info["macro"]:
            assert abs(info["macro"][metric] - np.mean([r[metric] for r in matches])) < 1e-12
    expected_epoch = max(cfg["epoch_candidates"], key=lambda e: (np.mean([r["recall@1"] for r in rows if r["condition"] == "real_eeg" and r["epoch"] == e]), -e))
    assert epoch == expected_epoch
    access = read(RUN / "access_summary.json")
    assert access["word_cache_members_read"] == ["train_words", "train_offsets"]
    assert access["structural_members_read"] == ["count_duration_only_train"]
    assert not any(access[k] for k in ("test_arrays_loaded", "test_evaluated", "outer_evaluated", "raw_source_files_opened"))
    protected = read(RUN / "preservation_before.json")
    assert all(sha(ROOT / key) == value for key, value in protected.items())
    assert read(RUN / "preservation_after.json")["unchanged"]
    write(RUN / "prediction_concentration.json", prediction_stats)
    final = {"passed": True, "folds": 8, "scored_rows": 152, "saved_model_replays": 48,
             "selected_query_ranks_reproduced": 2245 * 6, "initial_weights_match_cosine_baseline": True,
             "preprocessing_matches_cosine_baseline": True, "structural_predictions_match_baseline": True,
             "prior_protected_files_unchanged": len(protected), "outer_validation_or_test_evaluated": False}
    write(RUN / "final_verification.json", final)
    print(json.dumps(final, indent=2))
    archive = ROOT / "docs/research_results/gru_retrieval_loss_v1_20261101"
    if archive.exists():
        raise FileExistsError("Preserve existing report")
    archive.mkdir()
    old_summary = read(BASE / "training_cv_summary.json")
    old_scores = read(BASE / "training_cv_all_scores.json")
    audit = read(AUDIT / "summary.json")
    decision = read(RUN / "decision.json")
    lines = ["# GRU full-candidate retrieval-loss comparison — training CV only", "",
        "This is one fixed loss-only comparison against the preserved cosine GRU baseline. "
        "All scores are selected equal-reader training-CV results on eight original training readers, "
        "2,245 observations and 300 candidates; no outer validation or test evaluation occurred.", "",
        "## Diagnostic motivation", "",
        f"Replaying all 48 selected baseline checkpoints reproduced every saved rank. With real EEG, "
        f"the cosine GRU chose {audit['real_eeg']['unique_top1_sentences']:.3f} distinct top-1 sentences "
        f"per reader on average, and its largest top-1 share averaged "
        f"{100*audit['real_eeg']['largest_top1_share']:.2f}%. Mean cosine to the original training "
        f"target centroid was {audit['real_eeg']['mean_cosine_to_training_target_centroid']:.6f}. "
        "This is observed prediction concentration, not a proven cause or a pairing bug. "
        "Weak input information can also produce concentration; spreading predictions does not by "
        "itself demonstrate useful EEG information.", "",
        "## Fixed comparison", "",
        "The 45,952-parameter 841→32 projection, 32-unit GRU and 32→384 output head are unchanged. "
        "All original model initial tensors and preprocessing arrays match the cosine run exactly. "
        "Data order, masks, word counts, source word ordering, target embeddings, noise draws, "
        "optimizer (AdamW, LR .001, decay .01), batch size 64, clipping 1, seed 42, 12-epoch budget "
        "and checkpoints 4/8/12 remain fixed. No additional hyperparameter search occurred.", "",
        "Only the neural loss changes: instead of positive-only mean(1−cosine), normalized predicted "
        "embeddings score every unique fold-training candidate, divided by fixed temperature .07, "
        "and cross-entropy supervises the correct class index. Repeated target observations do not "
        "duplicate candidates or create false negatives. Each fold has all 300 training classes; "
        "this is checked before fitting. Word text/identity never enters the EEG feature matrix.", "",
        "The common epoch is selected by real-EEG macro training-CV Recall@1 from 4/8/12, with "
        "earlier epochs winning ties; every neural control uses that epoch. This procedure gives "
        "real EEG selection opportunity and produces optimistic development scores. Controls are "
        "not independently epoch-optimized. Both objective runs use the same rule. The report also "
        "compares the old objective at the new objective's selected epoch to avoid presenting an "
        "epoch difference as a loss effect. No outer arrays are read regardless of the gate.", "",
        f"Cosine selected epoch: **{base_epoch}**. Full-candidate CE selected epoch: **{epoch}**.", "",
        "## Full-candidate CE selected training-CV results", "",
        "Recall is percent; MRR is a reciprocal-rank score, not accuracy.", "",
        "| Condition | Macro R@1 % | Macro R@5 % | Macro R@10 % | Macro MRR |",
        "|---|---:|---:|---:|---:|"]
    for c, row in summary.items():
        m = row["macro"]
        lines.append(f"| {c} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |")
    lines += ["| Uniform ranking expectation | 0.3333 | 1.6667 | 3.3333 | 0.020942 |", "",
        "## Objective comparison for real EEG", "",
        "| Evaluation point | Macro R@1 % | Macro MRR |", "|---|---:|---:|"]
    matched = [r for r in old_scores if r["condition"] == "real_eeg" and r["epoch"] == epoch]
    lines += [f"| Cosine at its selected epoch {base_epoch} | {100*old_summary['real_eeg']['macro']['recall@1']:.4f} | {old_summary['real_eeg']['macro']['mrr']:.6f} |",
              f"| Cosine at matched epoch {epoch} | {100*np.mean([r['recall@1'] for r in matched]):.4f} | {np.mean([r['mrr'] for r in matched]):.6f} |",
              f"| Full-candidate CE at selected epoch {epoch} | {100*summary['real_eeg']['macro']['recall@1']:.4f} | {summary['real_eeg']['macro']['mrr']:.6f} |", "",
              "## Real minus comparators", "",
              "| Comparator | ΔR@1 pp | ΔMRR | Readers with positive R@1 gap |", "|---|---:|---:|---:|"]
    gaps = read(RUN / "real_minus_controls.json")
    for c, gap in gaps.items():
        positive = sum(v["recall@1"] > 0 for v in gap["per_reader_difference"].values())
        lines.append(f"| {c} | {100*gap['macro_difference']['recall@1']:+.4f} | {gap['macro_difference']['mrr']:+.6f} | {positive}/8 |")
    real_stats = [r for r in prediction_stats if r["condition"] == "real_eeg"]
    lines += ["", "## Prediction concentration and decision", "",
        f"New real-EEG outputs choose {np.mean([r['unique_top1_sentences'] for r in real_stats]):.3f} "
        f"distinct top-1 sentences per reader on average; the largest top-1 share averages "
        f"{100*np.mean([r['largest_top1_share'] for r in real_stats]):.2f}%. "
        "Interpret this together with control gaps, not as success by itself.", "",
        f"Prespecified descriptive gate passed: **{decision['training_gate_passed']}**. The inherited "
        "gate requires at least +2 pp R@1, +.005 MRR, and positive R@1 differences in six of eight "
        "readers against every comparator, including structure. It is not a significance test. "
        "Stop after this bounded training-only run; no search expansion or outer/test evaluation.", "",
        "## Verification and limitations", "",
        f"Twenty focused tests passed before fitting. Final checks replayed all 48 selected models "
        f"and reproduced 13,470 ranks; initialization and all preprocessing arrays match baseline, "
        f"and the structural comparator predictions match exactly. {len(protected)} prior files "
        "remain SHA256-identical; raw dataset size/time inventory is unchanged. Existing code changes, "
        "datasets, targets, checkpoints, results, main report, presentation and .venv are preserved.", "",
        "One loss, temperature and seed were tested. Overlapping training folds and repeated "
        "sentences are not independent cohorts. These readers have extensive development history. "
        "Source fixation/stimulus-assisted alignment, structural shortcuts, upstream preprocessing "
        "and provenance limits remain. The Gaussian null preserves masks/counts and raw marginal "
        "moments but not full real-data covariance/distribution. This comparison does not establish "
        "unseen-sentence generalization or a validated brain-to-text decoder.", "",
        "Negative-candidate training is a general representation-learning idea; "
        "[SimCSE](https://aclanthology.org/2021.emnlp-main.552/) studies contrastive text embeddings. "
        "This EEG experiment is not a reproduction of its model or performance.", "",
        "## Per-reader selected CE results", "",
        "| Condition | Training holdout reader | R@1 % | R@5 % | R@10 % | MRR |",
        "|---|---|---:|---:|---:|---:|"]
    for c, row in summary.items():
        for reader, m in row["per_reader"].items():
            lines.append(f"| {c} | {reader} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |")
    lines += ["", "Full histories, scored epochs, candidate pools, fold identities, controls, configurations "
              "and source snapshots are archived here. Coefficients/checkpoints remain in the new run folder."]
    (archive / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = []
    sources = (list(RUN.glob("*.json")) + list(RUN.glob("implementation/*"))
               + list(RUN.glob("cv_*/*.json")) + list(RUN.glob("cv_*/*/history.json"))
               + [Path(__file__), ROOT / "scripts/audit_gru_predictions.py"])
    for source in sources:
        rel = source.relative_to(RUN) if RUN in source.parents else Path("reporting") / source.name
        dest = archive / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest.append({"source": str(source.relative_to(ROOT)), "archive": str(rel), "sha256": sha(source)})
    for source in AUDIT.glob("*.json"):
        rel = Path("baseline_prediction_audit") / source.name
        dest = archive / rel
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
        manifest.append({"source": str(source.relative_to(ROOT)), "archive": str(rel), "sha256": sha(source)})
    write(archive / "archive_manifest.json", manifest)
    print(archive / "README.md")


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        main()
