"""Replay frozen GRU checkpoints to diagnose concentration; no model fitting."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from eeg_decoding.data.dataset import read_manifest, observation_seed
from eeg_decoding.training.alignment_train import TextCache
from eeg_decoding.evaluation.gru_pilot import SequenceGRU, encode, collate
from eeg_decoding.evaluation.spectral_pilot import matched_gaussian
from eeg_decoding.evaluation.retrieval import normalize, ranks, summarize
from run_gru_pilot import sha, write, ident, predict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/gru_pilot_v1_20261101"
OUTPUT = ROOT / "results/gru_prediction_audit_v1_20261101"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve prior audit")
    prior = json.loads((SOURCE / "plan.json").read_text())
    old = json.loads((ROOT / "results/word_order_pilot_v1_20261031/plan.json").read_text())
    records = {tuple(ident(r)): r for r in read_manifest(old["manifest"])}
    rows = [records[tuple(k)] for k in prior["train_observations"]]
    pool = prior["training_candidate_keys"]
    text = TextCache(old["text_cache"])
    with np.load(ROOT / "results/word_order_pilot_v1_20261031/word_sequences.npz", allow_pickle=False) as z:
        words, offsets = z["train_words"], z["train_offsets"]
    sequences = [words[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    by_id = {tuple(ident(r)): i for i, r in enumerate(rows)}
    cfg = prior["configuration"]
    epoch = json.loads((SOURCE / "frozen_selection.json").read_text())["common_epoch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    protected = {str(p.relative_to(ROOT)): sha(p) for p in SOURCE.rglob("*") if p.is_file()}
    OUTPUT.mkdir(parents=True)
    write(OUTPUT / "plan.json", {"scope": "Read-only replay of selected frozen checkpoints on original training CV readers",
          "epoch": epoch, "candidate_keys": pool, "training_observations": prior["train_observations"],
          "outer_validation_or_test_arrays_read": False, "model_training": False, "protected_sha256": protected})
    stats = []
    for folder in sorted(SOURCE.glob("cv_*")):
        provenance = json.loads((folder / "provenance.json").read_text())
        a = [by_id[tuple(k)] for k in provenance["fit_observations"]]
        b = [by_id[tuple(k)] for k in provenance["query_observations"]]
        query = [rows[i] for i in b]
        with np.load(folder / "preprocessing.npz", allow_pickle=False) as z:
            real = {k: z["real_" + k] for k in ("center", "scale")}
            noise_scaler = {k: z["gaussian_" + k] for k in ("center", "scale")}
            moments = {k: z["moments_" + k] for k in ("mean", "sd")}
        centroid = text.get([rows[i].key for i in a]).mean(0)
        centroid_direction = centroid / np.linalg.norm(centroid)
        constant_train_loss = 1 - float(np.linalg.norm(centroid))
        original_scores = json.loads((folder / "scores.json").read_text())
        for condition in cfg["conditions"]:
            scaler = noise_scaler if condition == "matched_gaussian" else real
            query_words = [sequences[i] for i in b]
            if condition == "matched_gaussian":
                query_words = [matched_gaussian(w, moments, observation_seed(r, cfg["seed"])) for w, r in zip(query_words, query)]
            encoded = [encode(w, scaler, condition, observation_seed(r, cfg["seed"] + 1)) for w, r in zip(query_words, query)]
            model = SequenceGRU(384, cfg["projection"], cfg["hidden"]).to(device)
            checkpoint = torch.load(folder / condition / f"epoch_{epoch}.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            prediction = predict(model, encoded, device, cfg["batch_size"])
            rank = ranks(prediction, text.get(pool), [r.key for r in query], pool)
            saved = next(r for r in original_scores if r["condition"] == condition and r["epoch"] == epoch)
            np.testing.assert_array_equal(rank, saved["ranks"])
            unit = normalize(prediction)
            scores = unit @ normalize(text.get(pool)).T
            winners = scores.argmax(1)
            counts = Counter(winners.tolist())
            probs = np.array(list(counts.values())) / len(winners)
            history = json.loads((folder / condition / "history.json").read_text())
            row = {"reader": folder.name[3:], "condition": condition, "metrics": summarize(rank),
                   "unique_top1_sentences": len(counts), "largest_top1_share": float(max(probs)),
                   "effective_top1_sentences": float(np.exp(-np.sum(probs * np.log(probs)))),
                   "mean_cosine_to_training_target_centroid": float((unit @ centroid_direction).mean()),
                   "normalized_prediction_mean_norm": float(np.linalg.norm(unit.mean(0))),
                   "mean_coordinate_sd": float(unit.std(0).mean()),
                   "constant_centroid_training_loss": constant_train_loss,
                   "observed_training_loss_at_selected_epoch": history[epoch - 1]["training_cosine_loss"],
                   "winner_counts": {pool[k]: v for k, v in counts.items()}}
            stats.append(row)
        write(OUTPUT / "fold_prediction_statistics.json", stats)
        print(folder.name, "replayed; real unique/top1 share", stats[-6]["unique_top1_sentences"], stats[-6]["largest_top1_share"], flush=True)
    summary = {c: {key: float(np.mean([r[key] for r in stats if r["condition"] == c]))
                   for key in ("unique_top1_sentences", "largest_top1_share", "effective_top1_sentences",
                               "mean_cosine_to_training_target_centroid", "normalized_prediction_mean_norm",
                               "constant_centroid_training_loss", "observed_training_loss_at_selected_epoch")}
               for c in cfg["conditions"]}
    write(OUTPUT / "summary.json", summary)
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    write(OUTPUT / "verification.json", {"saved_model_replays": len(stats), "all_selected_ranks_reproduced": True,
          "protected_files_unchanged": len(protected), "no_fitting": True, "outer_validation_or_test_read": False})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        main()
