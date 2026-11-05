"""Read and summarize a completed frozen word-order pilot; never fit models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main(run):
    plan = json.loads((run / "plan.json").read_text())
    reports = json.loads((run / "validation_summary.json").read_text())
    decision = json.loads((run / "decision.json").read_text())
    coverage = json.loads((run / "feature_view.json").read_text())["coverage"]
    preservation = json.loads((run / "preservation_after.json").read_text())
    assert preservation["unchanged"] and not decision["test_evaluated"]
    assert len(reports) == 7
    for r in reports:
        assert r["evaluation_split"] == "val" and r["metrics"]["n_queries"] == 512
        assert r["candidate_keys"] == plan["candidate_keys"] and len(r["candidate_keys"]) == 298
        assert r["observation_identities"] == plan["validation_observations"]
    folder = ROOT / "docs/research_results/word_order_pilot_v1_20261031"
    if folder.exists():
        raise FileExistsError("Preserve existing report")
    folder.mkdir()
    by = {r["condition"]: r for r in reports}
    lines = ["# Ordered word-level TRT pilot — validation only", "",
        f"**No useful improvement beyond controls; stopped without test evaluation.** Ordered real "
        f"Recall@1 is {100*by['ordered_real']['metrics']['recall@1']:.4f}%, versus "
        f"{100*by['mean_real']['metrics']['recall@1']:.4f}% for the matched mean baseline, "
        f"{100*by['matched_gaussian']['metrics']['recall@1']:.4f}% for Gaussian and "
        f"{100*by['structural_only']['metrics']['recall@1']:.4f}% for structure. "
        "The mean-to-ordered difference is one extra correct validation query, not evidence of success.", "",
        "This pilot tests whether retaining source word order improves over averaging word EEG. "
        "It is a small linear pilot, not a newly trained Transformer or a claimed decoding success.", "",
        "## Fixed protocol and selection", "",
        "Same ZuCo1 seed-42 cohort: 2,245 training observations from eight readers; "
        "512 reused validation queries from ZJS/ZKW, with the same 298 sentence candidates. "
        "No target correction, added observation or new exclusion. No test-source files, test features "
        "or test scores are used. The two validation readers are not independent replicated cohorts.", "",
        "TRT supplies eight bands × 105 ordered source channels per word. The bounded reader checks "
        "parent sentence pairing and skips word text, raw eye-tracking traces, pupil and gaze coordinates. "
        "Word identity, sentence ID and subject ID never enter numeric model matrices. The word axis is "
        "source stimulus order, not fixation chronology or inferred physical timing.", "",
        "For each held-out training reader, standardization uses finite words from the remaining seven "
        "readers only. A deterministic sample of at most 8,192 of these words fits 16-component randomized "
        "PCA without whitening. Four disjoint word-position bins each average available projected words; "
        "empty bins use zero. Four valid-word fractions are appended (68 features). Mean-real uses the "
        "same projection and a single bin (17 features). A sentence-level median imputer and standard "
        "scaler fit training rows only. These are feature widths, not matched model capacities; the "
        "same-width word-order-shuffled control is needed to assess order.", "",
        "The historical spectral mean baseline used 891 inputs including explicit structure and no "
        "word-level PCA. Its 12.3047% selected validation Recall@1 is preserved, but is not a matched "
        "baseline for these 17/68-input models. The earlier 17.7734% structural score was selected on "
        "outer validation MRR; the new structural setting is selected on inner-training Recall@1. "
        "Differences from those historical scores must not be attributed solely to word order. "
        "Failure of this bounded PCA/pooling pilot does not establish that every word-sequence model fails.", "",
        "Each arm selects one of three ridge alphas [1, 10, 100] using **equal-reader mean Recall@1 "
        "over all eight leave-one-training-reader-out folds**. Ties prefer stronger regularization. "
        "All Gaussian moments, PCA fits, scaling and count bases are refitted inside each fold. "
        "Candidate sentence embeddings are targets/public candidates, never EEG inputs. After selection "
        "is saved, preprocessing/model parameters refit on the original eight training readers and each "
        "frozen arm is evaluated once on the outer validation cohort. Outer validation does not select "
        "alphas, epochs or transforms. This cohort has already informed earlier studies, so this remains "
        "exploratory and cannot be described as fresh independent confirmation.", "",
        "## Controls and leakage limitations", "",
        "Gaussian words preserve exact word counts and finite/absent positions, using training-word "
        "column means and SDs; moments and Gaussian PCA are independently fitted per training fold. "
        "Word-order shuffling permutes complete EEG vectors among complete slots within each sentence, "
        "retaining mask positions and the mean. Class shuffling deranges training sentence labels "
        "consistently across repeated observations. Missingness-only uses four bin fractions, word count, "
        "overall valid fraction and a training-derived count basis. Structural-only reuses the prior "
        "14 count/duration columns and rebuilds its count basis within each fold. Both are non-amplitude "
        "controls. Gaussian and real arms have identical widths and alpha budgets.", "",
        "All word alignment remains gaze/stimulus assisted; fractions and bin boundaries can reveal "
        "reading structure and word-count information. These results must not be called annotation-free "
        "EEG decoding. Duration values are source sample counts at 500 Hz and enter only structural "
        "controls. The prior source audit's channel/band definitions, gamma2 code/descriptor discrepancy, "
        "unverified exact producer revision, source-session preprocessing and boundary limitations "
        "remain: [feature feasibility](../../word_eeg_feasibility_20261030.md).", "",
        "## Coverage", "", "| Split | Observations | Word slots | Complete EEG words | All-missing observations |",
        "|---|---:|---:|---:|---:|"]
    for c in coverage:
        lines.append(f"| {c['split']} | {c['observations']} | {c['words']} | {c['complete_words']} | {len(c['all_missing_observations'])} |")
    lines += ["", "All-missing training observation ZMG/70 is retained; no validation observation is removed.", "",
        "## Frozen selected validation scores", "", "Recall is percent. MRR is not accuracy.", "",
        "| Condition | Alpha (training CV) | Features | R@1 % | R@5 % | R@10 % | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        m = r["metrics"]
        lines.append(f"| {r['condition']} | {r['selection']['alpha']:g} | {r['n_features']} | "
                     f"{100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |")
    lines.append("| Uniform random ranking | — | — | 0.3356 | 1.6779 | 3.3557 | 0.021060 |")
    lines += ["", "## Ordered real minus comparators", "", "| Comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR |",
              "|---|---:|---:|---:|---:|"]
    for gap in json.loads((run / "real_minus_controls.json").read_text()):
        lines.append(f"| {gap['comparison']} | {100*gap['recall@1']:+.4f} | {100*gap['recall@5']:+.4f} | "
                     f"{100*gap['recall@10']:+.4f} | {gap['mrr']:+.6f} |")
    lines += ["", "## Reader-level results", "", "| Condition | Validation reader | Queries | R@1 % | MRR |",
              "|---|---|---:|---:|---:|"]
    for r in reports:
        for reader, m in r["per_subject"].items():
            lines.append(f"| {r['condition']} | {reader} | {m['n_queries']} | {100*m['recall@1']:.4f} | {m['mrr']:.6f} |")
    lines += ["", "## Prespecified descriptive screen", "",
        "Before fitting, the screen required ordered real to exceed the mean baseline and every "
        "control on aggregate Recall@1 and MRR, and to exceed Gaussian and structural Recall@1 for "
        "each validation reader. This is a descriptive stopping rule, not a significance test.", "",
        f"Screen passed: **{decision['descriptive_screen_passed']}**.", ""]
    for key, value in decision["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "**Stop after this validation-only pilot. No additional search or test evaluation.** "
              "A higher single score cannot establish robust EEG benefit or predict future accuracy.", "",
              "## Preservation and software verification", "",
              f"All {preservation['protected_files']} protected prior artifacts match pre-run SHA256 hashes. "
              "Existing data, checkpoints, previous results, report and presentation are preserved. "
              "Eleven focused tests passed before the run (word-order and existing spectral-pilot tests). "
              "This is not a new full-suite test count.", "",
              "Full fold scores, train-reader provenance, frozen selections, preprocessing, coefficients, "
              "candidate pools, per-query ranks and source hashes remain in "
              "`results/word_order_pilot_v1_20261031`. JSON evidence and source snapshots are archived here."]
    (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = []
    sources = list(run.glob("*.json")) + list(run.glob("*/validation.json")) + list(run.glob("implementation/*")) + [Path(__file__)]
    for source in sources:
        rel = source.relative_to(run) if run in source.parents else Path("reporting") / source.name
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        manifest.append({"source": str(source.relative_to(ROOT)), "archive": str(rel),
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    write_path = folder / "archive_manifest.json"
    write_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(folder / "README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run.resolve())
