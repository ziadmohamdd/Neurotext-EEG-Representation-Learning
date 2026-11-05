"""Check stored fold scores and selection arithmetic; no model fitting."""
import json
from pathlib import Path
import numpy as np
from eeg_decoding.evaluation.retrieval import summarize
from eeg_decoding.evaluation.relative_trt import choose
from run_amplitude_audit import sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/amplitude_audit_v1_20261101"


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    rows = json.loads((RUN / "inner_training_scores.json").read_text())
    summary = json.loads((RUN / "training_summary.json").read_text())
    trainids = {tuple(k) for k in plan["train_observations"]}
    assert len(trainids) == 2245 and len(rows) == 576 and len(summary) == 12
    assert not {k[1] for k in trainids} & {"ZJS", "ZKW", "ZAB", "ZKB"}
    assert plan["word_sequence_arrays_read"] == ["train_words", "train_offsets"]
    assert plan["structural_arrays_read"] == ["count_duration_only_train"]
    for reader in plan["readers"]:
        fold = json.loads((RUN / ("fold_" + reader + "_provenance.json")).read_text())
        fitids = {tuple(k) for k in fold["fit_observations"]}
        queryids = {tuple(k) for k in fold["query_observations"]}
        assert fitids | queryids == trainids and not fitids & queryids
        assert {k[1] for k in queryids} == {reader} and reader not in {k[1] for k in fitids}
        assert fold["candidate_keys"] == plan["candidate_keys"] and fold["gaussian_masks_match"]
        mapping = fold["class_shuffle_mapping"]
        assert len(mapping) == len(set(mapping.values())) and all(k != v for k, v in mapping.items())
        for row in [r for r in rows if r["reader"] == reader]:
            assert row["n_queries"] == len(queryids)
            metrics = summarize(row["ranks"])
            for key in metrics:
                assert metrics[key] == row[key]
    for result in summary:
        assert choose(result["grid"]) == result["selection"]
        for entry in result["grid"]:
            matches = [r for r in rows if (r["head"], r["condition"], r["weight"], r["alpha"]) ==
                       (result["head"], result["condition"], entry["weight"], entry["alpha"])]
            assert len(matches) == 8
            for metric in ("recall@1", "recall@5", "recall@10", "mrr"):
                assert entry["macro_" + metric] == float(np.mean([r[metric] for r in matches]))
    for head in plan["configuration"]["heads"]:
        for reader in plan["readers"]:
            for alpha in plan["configuration"]["ridge_alphas"]:
                base = next(r for r in rows if (r["head"], r["reader"], r["condition"], r["alpha"]) ==
                            (head, reader, "structural_only", alpha))
                for condition in ("structural_real", "structural_gaussian"):
                    zero = next(r for r in rows if (r["head"], r["reader"], r["condition"], r["alpha"], r["weight"]) ==
                                (head, reader, condition, alpha, 0.))
                    assert zero["ranks"] == base["ranks"]
    old = json.loads((RUN / "preservation_before.json").read_text())
    assert all(sha(ROOT / name) == digest for name, digest in old.items())
    result = {"status": "passed", "score_rows_checked": len(rows), "selected_summaries_checked": len(summary),
              "reader_folds_checked": 8, "query_ranks_checked": sum(r["n_queries"] for r in rows),
              "zero_weight_predictions_match_structural": True,
              "prior_protected_files_unchanged": len(old), "outer_validation_extracted": False, "test_evaluated": False}
    path = RUN / "verification.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
