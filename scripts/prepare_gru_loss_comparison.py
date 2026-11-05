"""Create a new runner by making explicit, guarded edits to the frozen GRU source."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/gru_pilot_v1_20261101/implementation/run_gru_pilot.py"
TARGET = ROOT / "scripts/run_gru_retrieval_loss.py"
CONFIG = ROOT / "configs/gru_retrieval_loss_v1_20261101.json"


def main():
    if TARGET.exists() or CONFIG.exists():
        raise FileExistsError("Preserve generated source/configuration")
    text = BASE.read_text(encoding="utf-8")
    baseline_hash = hashlib.sha256(BASE.read_bytes()).hexdigest()
    current = ROOT / "scripts/run_gru_pilot.py"
    if hashlib.sha256(current.read_bytes()).hexdigest() != baseline_hash:
        raise ValueError("Current baseline differs from the archived experimental source")
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError("Unexpected baseline text: " + old[:90])
        text = text.replace(old, new)
    replace('"""One frozen GRU pilot. Only training cache members are read before gate."""',
            '"""Loss-only GRU comparison; outer validation and test arrays are never loaded."""\nimport os\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")')
    replace("CONFIG = ROOT / 'configs/gru_pilot_v1_20261101.json'", "CONFIG = ROOT / 'configs/gru_retrieval_loss_v1_20261101.json'")
    replace("from eeg_decoding.evaluation.controls import shuffled_keys", "from eeg_decoding.evaluation.controls import shuffled_keys\nfrom eeg_decoding.evaluation.gru_retrieval_loss import candidate_cross_entropy")
    replace("    keys = [r.key for r in train]; wrong = shuffled_keys(keys, cfg['seed'])",
            "    keys = [r.key for r in train]; wrong = shuffled_keys(keys, cfg['seed'])\n"
            "    training_pool = sorted(set(keys))\n"
            "    assert training_pool == pool, 'All candidates must be unique fold-training classes'\n"
            "    lookup = {key:i for i,key in enumerate(training_pool)}\n"
            "    candidate_vectors = torch.as_tensor(text.get(training_pool), dtype=torch.float32, device=device)")
    replace("        targets = torch.as_tensor(text.get(wrong if condition=='class_shuffled' else keys), dtype=torch.float32, device=device)",
            "        target_keys = wrong if condition=='class_shuffled' else keys\n"
            "        labels = torch.as_tensor([lookup[k] for k in target_keys], dtype=torch.long, device=device)\n"
            "        targets = torch.as_tensor(text.get(target_keys), dtype=torch.float32, device=device)")
    replace("                loss=(1-torch.nn.functional.cosine_similarity(model(x,n),targets[ix],dim=1)).mean()",
            "                loss=candidate_cross_entropy(model(x,n),candidate_vectors,labels[ix],cfg['temperature'])")
    replace("history.append({'epoch':epoch,'training_cosine_loss':total/len(train)})",
            "history.append({'epoch':epoch,'training_candidate_ce':total/len(train)})")
    replace("'authorization':'Separately versioned GRU pilot explicitly authorized 2026-11-01; test prohibition remains'",
            "'authorization':'User requested an alternative; prediction-concentration audit motivated a loss-only comparison; no outer validation or test evaluation'")
    replace("ROOT/'src/eeg_decoding/evaluation/incremental_ridge.py']:",
            "ROOT/'src/eeg_decoding/evaluation/incremental_ridge.py',ROOT/'src/eeg_decoding/evaluation/gru_retrieval_loss.py',ROOT/'tests/test_gru_retrieval_loss.py',ROOT/'scripts/prepare_gru_loss_comparison.py']:")
    replace("'outer_validation_permitted':passed", "'outer_validation_permitted':False")
    start = text.index("    # The gate and frozen selection are persisted")
    stop = text.index("    changed=[p for p,h in before.items()", start)
    text = text[:start] + "    # This bounded loss-only experiment stops on training CV, regardless of gate.\n" + text[stop:]
    replace("['train_words','train_offsets']+(['val_words','val_offsets'] if passed else [])", "['train_words','train_offsets']")
    replace("['count_duration_only_train']+(['count_duration_only_val'] if passed else [])", "['count_duration_only_train']")
    replace("'outer_evaluated':passed", "'outer_evaluated':False")
    cfg = json.loads((ROOT / "configs/gru_pilot_v1_20261101.json").read_text())
    cfg.update({"version": "gru_full_candidate_ce_loss_only_v1", "temperature": .07,
                "loss": "Cross entropy over all unique fold-training sentence embeddings, normalized cosine logits / fixed temperature 0.07",
                "baseline_runner_sha256": baseline_hash, "baseline": "results/gru_pilot_v1_20261101",
                "hypothesis_evidence": "results/gru_prediction_audit_v1_20261101/summary.json",
                "hypothesis": "Positive-only cosine GRU outputs concentrate on few candidates; explicit negatives may improve discrimination. Not a confirmed implementation bug or guaranteed EEG benefit.",
                "test_policy": "Training CV only; do not read outer validation or test feature arrays, even if the descriptive gate passes.",
                "comparison": "Only loss differs; same input/scaler, architecture, initial weights, orders, optimizer, 12-epoch budget and 4/8/12 common-epoch selection. Report cosine baseline at both its own selected epoch and CE-selected matched epoch.",
                "stopping": "One fixed temperature; no search expansion. Stop after training CV and archive all controls regardless of result."})
    TARGET.write_text(text, encoding="utf-8")
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(TARGET)
    print(CONFIG)


if __name__ == "__main__":
    main()
