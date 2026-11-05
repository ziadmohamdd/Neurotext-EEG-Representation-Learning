"""Archive the training-only amplitude/target audit without running any model."""
import json
from pathlib import Path
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/amplitude_audit_v1_20261101"


def main():
    summary = json.loads((RUN / "training_summary.json").read_text())
    decision = json.loads((RUN / "decision.json").read_text())
    audit = json.loads((RUN / "training_feature_audit.json").read_text())
    preserved = json.loads((RUN / "preservation_after.json").read_text())
    assert not decision["test_evaluated"] and not decision["outer_validation_extracted"] and preserved["unchanged"]
    destination = ROOT / "docs/research_results/amplitude_audit_v1_20261101"
    if destination.exists():
        raise FileExistsError("Preserve prior report")
    destination.mkdir()
    raw = list(audit["reader_raw_mean_norms"].values())
    relative = list(audit["reader_relative_mean_norms"].values())
    lines = ["# Full-width relative TRT and target-head audit — training readers only", "",
        "**These are selected inner-training-reader cross-validation results, not outer validation "
        "or test accuracy.** No outer validation/test array was extracted, no raw source MAT file "
        "was reopened, and no previous model or report was replaced. The fixed run has ended.", "",
        "## Diagnosed hypotheses", "",
        "The prior word-order pilot retained 75.2488% of standardized training-word variance in "
        "16 PCA components. Variance retention does not measure retained semantic information and "
        "does not prove PCA caused its failure. No confirmed pairing/ranking implementation bug "
        "was identified in the reviewed paths; earlier saved-model verification remains intact.", "",
        f"Original training-reader mean feature norms range from {min(raw):.4f} to {max(raw):.4f} "
        f"(ratio {max(raw)/min(raw):.4f}). The new transform gives {min(relative):.4f} to "
        f"{max(relative):.4f} (ratio {max(relative)/min(relative):.4f}). This is an amplitude "
        "diagnostic, not a demonstration of successful participant alignment or retrieval.", "",
        "The new representation uses all 840 supplied TRT coordinates (eight bands × 105 source "
        "channels), with no PCA. Complete words undergo signed log1p and subtraction of the "
        "105-channel mean separately within each band. For the observed nonnegative real values "
        "this is log1p; negative Gaussian draws use the same signed transform without clipping. "
        "Sentence features average the transformed complete words. All-missing ZMG/70 remains "
        "in training, handled by training-fitted imputation. This changes the representation and "
        "drops word order; it is not an isolated PCA-component ablation.", "",
        "## Frozen design and selection", "",
        "Only the original eight training readers and 2,245 observations enter this run. "
        "The fixed candidate vocabulary consists of their 300 unique sentence classes. Each fold "
        "holds one training reader out. Source word-cache arrays read are exactly `train_words` "
        "and `train_offsets`; structural input reads only `count_duration_only_train`. "
        "ZJS/ZKW outer validation and ZAB/ZKB test EEG are not used. Metadata manifests contain "
        "identities for provenance but only declared training rows are selected.", "",
        "Each fold fits Gaussian moments from the other seven readers' words, draws null words "
        "with original word counts/masks, then applies exactly the same relative transform. "
        "Gaussian values retain source-scale marginal moments before the nonlinear transform, "
        "not after it. A Gaussian null does not preserve the full positive real-feature distribution "
        "or covariance; superiority over it alone would not establish EEG semantics.", "",
        "Structural inputs are the prior 14 count/duration summaries plus a count basis rebuilt "
        "from fold-training counts. The structural median/standard transform is frozen and shared "
        "across structure-bearing conditions within a fold. Real and Gaussian median imputation/" 
        "standardization fit fold-training rows only. EEG weights apply after scaling. "
        "Durations are source sample counts at 500 Hz, not inferred onset times.", "",
        "Two heads use the same ridge design: (1) semantic sentence embeddings with cosine retrieval, "
        "and (2) 300 one-hot class targets ranked by raw class score. The latter is **closed-set "
        "classification of seen sentence classes**, not unrestricted text generation, unseen-text "
        "generalization, or proof of semantic understanding. Sentence IDs/word text are not numeric "
        "EEG inputs; class labels are supervision only. Gaze/stimulus-assisted source segmentation "
        "and missingness remain annotation information.", "",
        "Each head/combined arm has the same nine settings: alpha [1,10,100] × post-scaling EEG "
        "weight [0,0.1,1]. Structural-only and amplitude-only comparators have three alphas each. "
        "Selection maximizes equal-reader mean Recall@1; roundoff ties prefer zero/smaller EEG "
        "weight, then larger alpha. Combined arms have more search opportunity than structural-only; "
        "real and Gaussian search budgets match. Zero EEG weight exactly reuses structural predictions. "
        "The class-shuffled arm consistently deranges the entire training target mapping, including "
        "the structural association; it is a whole-pair control, not an isolated EEG-block shuffle.", "",
        "There are 576 scored fold/settings/head rows (36 settings per head × 2 heads × 8 readers). "
        "All settings are archived. Selected CV scores reuse the same eight readers for choice and "
        "description, so are optimistic exploration, not independent confirmation or significance "
        "tests. The prior studies also used these training participants. Results below must not be "
        "directly compared with historical 512-query/298-candidate validation accuracy.", "",
        "## Selected training-reader CV results", "",
        "Recall is equal-reader macro percent; MRR is not accuracy. Chance Recall@1 for 300 candidates "
        "is 0.3333%.", "",
        "| Head | Condition | Alpha | EEG weight | Macro R@1 % | Macro R@5 % | Macro R@10 % | Macro MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        s = r["selection"]
        lines.append(f"| {r['head']} | {r['condition']} | {s['alpha']:g} | {s['weight']:g} | "
                     f"{100*s['macro_recall@1']:.4f} | {100*s['macro_recall@5']:.4f} | "
                     f"{100*s['macro_recall@10']:.4f} | {s['macro_mrr']:.6f} |")
    lines += ["", "## Prespecified next-stage screen", "",
        "Before the run, a head's structural+real model had to exceed both structural-only and "
        "structural+Gaussian by at least two percentage points of macro Recall@1 and show positive "
        "reader differences to both on at least six of eight readers. This is a descriptive "
        "screen, not a significance test. The run stops without any outer evaluation regardless.", "",
        "| Head | Real − structural R@1 pp | Real − Gaussian-combined R@1 pp | Positive readers vs structural | Positive readers vs Gaussian | Passed |",
        "|---|---:|---:|---:|---:|---|"]
    for g in decision["gates"]:
        lines.append(f"| {g['head']} | {100*g['macro_recall_gaps']['structural_only']:+.4f} | "
                     f"{100*g['macro_recall_gaps']['structural_gaussian']:+.4f} | "
                     f"{g['positive_readers']['structural_only']}/8 | "
                     f"{g['positive_readers']['structural_gaussian']}/8 | {g['passed']} |")
    if not any(g["passed"] for g in decision["gates"]):
        lines += ["", "**Neither head passed.** Reducing reader amplitude differences and retaining all "
                  "channels did not establish a sufficiently consistent incremental EEG advantage. "
                  "No additional outer validation or test evaluation was justified or performed."]
    else:
        lines += ["", "A head passed the training-only screen. This is a candidate for a separately "
                  "frozen confirmation protocol, not a solved EEG decoder. No outer evaluation was performed."]
    lines += ["", "## Reader-level selected results", "",
              "| Head | Condition | Training holdout reader | Queries | R@1 % | MRR |",
              "|---|---|---|---:|---:|---:|"]
    for r in summary:
        for reader, m in r["per_reader"].items():
            lines.append(f"| {r['head']} | {r['condition']} | {reader} | {m['n_queries']} | "
                         f"{100*m['recall@1']:.4f} | {m['mrr']:.6f} |")
    lines += ["", "## Preservation, verification and literature context", "",
        f"All {preserved['protected_files']} prior protected files remained SHA256-identical, including "
        "previous results, reports, presentation, checkpoints and selected source caches. Eight focused "
        "tests passed for relative features, ties, selection and existing ridge-block invariants. "
        "No full-suite count is claimed for this run.", "",
        "The stored plan, all fold partitions and class mappings, every scored setting with query ranks, "
        "chosen settings, training-only audit, decision and implementation snapshots are archived here. "
        "The original word source cache remains untouched. This run stores CV evidence, not a newly "
        "validated deployable model; no model is selected using outer data.", "",
        "Related primary work studies word-level Conformer classification followed by sentence retrieval "
        "([Zhou et al., 2024](https://arxiv.org/abs/2408.04679)); this diagnostic is not a reproduction "
        "of its architecture, task or numerical results. Noise-controlled evaluation is motivated by "
        "[Jo et al., 2024](https://arxiv.org/abs/2405.06459), which found noise-equivalent performance "
        "for the EEG-to-text generators they evaluated; that result does not prove the same cause here."]
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = []
    for source in list(RUN.glob("*.json")) + list(RUN.glob("implementation/*")) + [Path(__file__)]:
        rel = source.relative_to(RUN) if RUN in source.parents else Path("reporting") / source.name
        dest = destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest.append({"source": str(source.relative_to(ROOT)), "archive": str(rel),
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    (destination / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(destination / "README.md")


if __name__ == "__main__":
    main()
