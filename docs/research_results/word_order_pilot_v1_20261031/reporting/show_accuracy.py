"""Display saved validation retrieval scores without fitting or evaluating models."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = {
    "incremental": ROOT / "results/trt_incremental_v1_20261030/validation_summary.json",
    "spectral": ROOT / "results/spectral_trt_pilot_v1_20261030/validation_summary.json",
    "word-order": ROOT / "results/word_order_pilot_v1_20261031/validation_summary.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=SUMMARIES, default="incremental")
    args = parser.parse_args()
    source = SUMMARIES[args.study]
    reports = json.loads(source.read_text(encoding="utf-8"))
    if not reports or any(r["evaluation_split"] != "val" for r in reports):
        raise ValueError("Expected saved validation-only results")
    candidates = reports[0]["candidate_keys"]
    observations = reports[0]["observation_identities"]
    if not candidates or any(
        r["candidate_keys"] != candidates or r["observation_identities"] != observations
        for r in reports
    ):
        raise ValueError("Conditions must share observations and candidate pools")
    print("VALIDATION ONLY - saved results; no training or test evaluation")
    print(f"Source: {source.relative_to(ROOT)}")
    print(f"Queries: {reports[0]['metrics']['n_queries']} | Candidates: {len(candidates)}")
    print("Top-1 accuracy = Recall@1. MRR is a ranking score, not accuracy.\n")
    print(f"{'Condition':<26} {'Recall@1':>10} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>10}")
    for report in reports:
        metrics = report["metrics"]
        recalls = " ".join(f"{100 * metrics[f'recall@{k}']:9.2f}%" for k in (1, 5, 10))
        print(f"{report['condition']:<26} {recalls} {metrics['mrr']:10.6f}")
    chance = " ".join(f"{100 * min(k, len(candidates)) / len(candidates):9.2f}%" for k in (1, 5, 10))
    print(f"{'Random chance':<26} {chance} {'--':>10}")
    print("\nRepeated validation use introduces selection optimism.")
    if args.study == "incremental":
        print("Each combined model searched 196 settings; structural-only searched 28.")
    elif args.study == "spectral":
        print("Each spectral pilot condition searched 28 settings.")
    else:
        print("Three alphas per arm; selected on eight training-reader holdouts only.")
        print("The reused outer validation cohort did not select settings in this pilot.")
    print("Conclusion: no consistent incremental real EEG advantage beyond controls.")


if __name__ == "__main__":
    main()
