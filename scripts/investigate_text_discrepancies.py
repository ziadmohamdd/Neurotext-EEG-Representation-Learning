"""Read-only ZuCo1 text audit. Writes only to a fresh investigation directory.

No target replacement, signal access, model imports, fitting, or evaluation.
The four primary categories are editorial judgements, not automatic normalization.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/text_discrepancy_investigation_20261030"
OLD = ROOT / "results/pairing_audit_20261030"
CATEGORIES = ("formatting_only", "canonical_key_difference", "substantive_wording", "unresolved_mapping")
GROUPS = {
    "formatting_only": [10, 23, 45, 51, 61, 91, 98, 100, 106, 122, 140, 150, 167, 180, 194, 235, 241, 248, 252, 295, 297],
    "canonical_key_difference": [26, 69, 71, 73, 114, 154, 251, 255],
    "substantive_wording": [24, 35, 64, 65, 66, 203, 227, 275],
    "unresolved_mapping": [],
}
NOTES = {
    10: "Spacing inside long-time; same intended wording.",
    24: "nVera-Ellen versus Vera-Ellen: extra letter in a name.",
    26: "Jose versus José: diacritic variant of same intended name.",
    35: "wrote versus write: different verb form; CSV is not automatically authoritative.",
    64: "Maria versus Marić: different surname spelling, not accent stripping.",
    65: "Maria versus Marić: different surname spelling, not accent stripping.",
    66: "Maria versus Marić: different surname spelling, not accent stripping.",
    69: "Eidgenoessische versus Eidgenössische: oe/ö transliteration.",
    71: "Wuerttemberg versus Württemberg: ue/ü transliteration.",
    73: "Baden-Wuerttemberg versus Baden-Württemberg: ue/ü transliteration.",
    98: "U+FFFD versus right apostrophe; presentation CSV supplies a corroborating punctuation interpretation.",
    100: "Two U+FFFD versus right apostrophes; presentation CSV corroborates.",
    114: "Zoonomia versus Zoönomia plus hyphen/en dash: diacritic and punctuation variants.",
    154: "Adams-Onis versus Adams-Onís: diacritic variant.",
    203: "km? versus km²: damaged unit exponent, not merely visual formatting.",
    227: "Extra 111 before Senator plus hyphen/en dash: substantive digit insertion.",
    248: "U+FFFD versus em dash in date range; presentation CSV corroborates.",
    251: "nee versus née: diacritic variant.",
    255: "Creteil versus Créteil: diacritic variant.",
    275: "Ruthy versus Réthy changes u to é; also Leopold/Léopold and date punctuation.",
}
REPAIR_PROPOSAL_IDS = {98, 100, 248}


def key(text):
    return hashlib.sha256(" ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode()).hexdigest()


def identity(record):
    return tuple(str(record[k]) for k in ("dataset", "subject", "session", "task", "sentence_id"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify(sentence_id):
    matches = [name for name, ids in GROUPS.items() if sentence_id in ids]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one reviewed category for {sentence_id}")
    return matches[0]


def count_rows(records, category_by_id, selected=None):
    result = {}
    for category in CATEGORIES:
        affected = [r for r in records if category_by_id.get(int(r["sentence_id"])) == category
                    and (selected is None or int(r["sentence_id"]) in selected)]
        result[category] = {"unique_sentences": len({key(r["text"]) for r in affected}), "observations": len(affected)}
    result["total"] = {"unique_sentences": sum(v["unique_sentences"] for v in result.values()),
                       "observations": sum(v["observations"] for v in result.values())}
    result["all_observations"] = len(records)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_mapping(annotation, presentation, sentences, by_id):
    if len(annotation) != 300 or len(presentation) != 595 or len(sentences) != 595:
        raise ValueError("Unexpected source sizes; review mapping before proceeding")
    if len({(r["sentence_id"], r["paragraph_id"]) for r in annotation}) != 300:
        raise ValueError("Annotation composite identifiers are not unique")
    for i, row in enumerate(annotation):
        other = presentation[i + 5]
        if not (row["sentence_id"] == other["paragraph_id"] and row["paragraph_id"] == other["sentence_id"]
                and row["control"] == other["control"] and by_id[i] == str(sentences[i + 5])):
            raise ValueError(f"Source alignment failed at export ID {i}")


def main(output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    prepared_path = ROOT / "data/processed/zuco1_batch/combined.jsonl"
    cohort_path = ROOT / "data/manifests/zuco1_duration_qc.jsonl"
    split_path = ROOT / "results/cnn_candidate_comparison_20261030/batch_real/splits.json"
    cache = ROOT / "data/processed/zuco1_batch/sentence_embeddings_a01b38226d177cab.npz"
    # Preserve all previous result metadata and checkpoints, plus current inputs.
    protected = sorted(set([prepared_path, cohort_path, cache] + [p for p in (ROOT / "results").rglob("*")
                       if p.is_file() and SOURCE not in p.parents]))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    inputs = [Path(__file__), ROOT / "tests/test_text_discrepancy_audit.py", prepared_path, cohort_path, split_path, cache, OLD / "official_relations_labels_task2.csv",
              OLD / "official_firstLevel_NR.m", OLD / "audit.json", OLD / "stimulus_variant_audit.json",
              SOURCE / "sentencesNR.mat", SOURCE / "relations_normal_reading.csv",
              SOURCE / "nemar_relations_labels_task2.tsv", SOURCE / "osf_exports_inventory.json",
              SOURCE / "osf_preprocessed_inventory.json", OLD / "osf_materials_inventory.json",
              OLD / "osf_matlab_inventory.json", ROOT / "data/raw/zuco1/download_plan.json"]
    prepared, cohort = read_jsonl(prepared_path), read_jsonl(cohort_path)
    by_id = {}
    for r in prepared:
        i = int(r["sentence_id"])
        if i in by_id and by_id[i] != r["text"]:
            raise ValueError(f"Conflicting exported targets at {i}")
        by_id[i] = r["text"]
    sentences = loadmat(SOURCE / "sentencesNR.mat", simplify_cells=True)["sentences"]
    with (OLD / "official_relations_labels_task2.csv").open(encoding="utf-8-sig", newline="") as f:
        annotation = list(csv.DictReader(f))
    # Encoding is inferred from coherent punctuation, NOT declared by the publisher.
    with (SOURCE / "relations_normal_reading.csv").open(encoding="mac_roman", newline="") as f:
        presentation = list(csv.DictReader(f, delimiter=";"))
    with (SOURCE / "nemar_relations_labels_task2.tsv").open(encoding="utf-8-sig", newline="") as f:
        mirror = list(csv.DictReader(f, delimiter="\t"))
    check_mapping(annotation, presentation, sentences, by_id)
    changed = {i for i in range(300) if by_id[i] != annotation[i]["sentence"]}
    reviewed = [i for ids in GROUPS.values() for i in ids]
    if len(reviewed) != len(set(reviewed)) or changed != set(reviewed):
        raise ValueError("Reviewed classification does not cover source differences exactly")
    category_by_id = {i: classify(i) for i in changed}
    split = json.loads(split_path.read_text())
    if split["seed"] != 42 or split["protocol"] != "subject":
        raise ValueError("Unexpected saved split")
    split_lookup = {}
    for name, details in split["splits"].items():
        for obs in details["observations"]:
            obs = tuple(obs)
            if obs in split_lookup:
                raise ValueError("Split overlap")
            split_lookup[obs] = name
    if set(split_lookup) != {identity(r) for r in cohort} or len(split_lookup) != len(cohort):
        raise ValueError("Saved split does not cover cohort exactly")
    cohort_ids = {identity(r) for r in cohort}
    counts = {"prepared": count_rows(prepared, category_by_id), "duration_qc_retained": count_rows(cohort, category_by_id)}
    repair_counts = {}
    for name in ("train", "validation", "test"):
        # Saved artifact uses 'val', not a regenerated random split.
        saved_name = "val" if name == "validation" and "val" in split["splits"] else name
        subset = [r for r in cohort if split_lookup[identity(r)] == saved_name]
        counts[name] = count_rows(subset, category_by_id)
        repair_counts[name] = count_rows(subset, category_by_id, REPAIR_PROPOSAL_IDS)["total"]
    ledger = []
    for i in sorted(changed):
        observed = [r for r in prepared if int(r["sentence_id"]) == i]
        row = {"export_sentence_id_zero_based": i, "export_MAT_slot_one_based": i + 1,
               "annotation_data_row_one_based": i + 1, "presentation_data_row_one_based": i + 6,
               "sentencesNR_MAT_index_one_based": i + 6, "category": category_by_id[i],
               "reason": NOTES.get(i, "ASCII hyphen versus en dash in a date range; same wording."),
               "export_text": by_id[i], "source_sentence_MAT_text": str(sentences[i + 5]),
               "annotation_text": annotation[i]["sentence"], "presentation_text_mac_roman_inferred": presentation[i + 5]["sentence"],
               "annotation_sentence_id": annotation[i]["sentence_id"], "annotation_paragraph_id": annotation[i]["paragraph_id"],
               "presentation_sentence_id": presentation[i + 5]["sentence_id"], "presentation_paragraph_id": presentation[i + 5]["paragraph_id"],
               "current_key": key(by_id[i]), "annotation_key": key(annotation[i]["sentence"]),
               "key_changes": key(by_id[i]) != key(annotation[i]["sentence"]),
               "punctuation_repair_proposed_only": i in REPAIR_PROPOSAL_IDS,
               "prepared_observations": len(observed), "qc_retained_observations": sum(identity(r) in cohort_ids for r in observed)}
        for name in ("train", "val", "test"):
            row[name + "_observations"] = sum(split_lookup.get(identity(r)) == name for r in observed)
        ledger.append(row)
    observations = [{"identity": list(identity(r)), "category": category_by_id[int(r["sentence_id"])],
                     "retained": identity(r) in cohort_ids, "split": split_lookup.get(identity(r))} for r in prepared if int(r["sentence_id"]) in changed]
    excluded = [list(identity(r)) for r in prepared if identity(r) not in cohort_ids]
    provenance = []
    for file in (SOURCE / "osf_preprocessed_inventory.json", SOURCE / "osf_exports_inventory.json", OLD / "osf_materials_inventory.json", OLD / "osf_matlab_inventory.json"):
        for item in json.loads(file.read_text())["data"]:
            a = item["attributes"]
            if a["kind"] == "file":
                provenance.append({"inventory": str(file.relative_to(ROOT)), "osf_file_id": item["id"],
                                   "path": a["materialized_path"], "version": a["current_version"],
                                   "modified": a["date_modified"], "sha256": a["extra"]["hashes"]["sha256"],
                                   "download": item["links"]["download"]})
    for filename in ("sentencesNR.mat", "relations_normal_reading.csv", "official_relations_labels_task2.csv"):
        local = (OLD if filename.startswith("official_") else SOURCE) / filename
        remote_name = filename.removeprefix("official_")
        matches = [p for p in provenance if p["path"].endswith("/" + remote_name)]
        if len(matches) != 1 or sha(local) != matches[0]["sha256"]:
            raise ValueError(f"Published source checksum mismatch: {filename}")
    plan = json.loads((ROOT / "data/raw/zuco1/download_plan.json").read_text())
    exports = [p for p in provenance if p["path"].split("/")[-1].startswith("results") and p["path"].endswith("_NR.mat")]
    pinned = {f["name"]: f["sha256"] for f in plan["files"] if f["role"] == "sentences"}
    if len(exports) != 12 or any(pinned[p["path"].split("/")[-1]] != p["sha256"] for p in exports):
        raise ValueError("Current export hashes differ from pinned download plan")
    summary = {"scope": "ZuCo1 normal reading; saved seed-42 CNN subject split only; not transfer splits",
               "training_launched": False, "targets_modified": False,
               "classification_version": "manual_v1_formatting_orthography_wording_mapping",
               "classifications": GROUPS, "current_export_hashes_match_pinned_plan": len(exports),
               "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
               "counts": counts, "punctuation_repair_proposal_counts": repair_counts,
               "duration_excluded_observations": excluded,
               "duration_excluded_discrepant_observations": [o for o in observations if not o["retained"]],
               "all_prepared_match_sentence_MAT_at_export_id_plus_5": all(r["text"] == str(sentences[int(r["sentence_id"]) + 5]) for r in prepared),
               "sentence_MAT_total_rows": len(sentences), "practice_rows": 5, "exported_sentence_rows": 300,
               "unused_trailing_sentence_MAT_rows": len(sentences) - 305,
               "crossed_identifier_matches": 300, "control_flag_matches": 300,
               "nemar_mirror_annotation_field_dicts_equal": annotation == mirror,
               "presentation_vs_sentence_MAT_different_zero_based_rows": [i for i, r in enumerate(presentation) if r["sentence"] != str(sentences[i])],
               "presentation_csv_encoding": "MacRoman inferred, not publisher-declared; bytes D1 and D5 interpreted as em dash and right apostrophe",
               "canonical_key_changed_discrepancies": sum(r["key_changes"] for r in ledger),
               "current_unique_candidate_keys": len({key(v) for v in by_id.values()}),
               "annotation_unique_candidate_keys": len({key(r["sentence"]) for r in annotation}),
               "decision": "Retain existing dataset; do not wholesale substitute annotation CSV. Propose separately versioned, limited punctuation repair for IDs 98,100,248, subject to explicit adoption; no corrected manifest created."}
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    if after != before:
        raise RuntimeError("Protected input changed during audit")
    output.mkdir(parents=True)
    for name, content in (("summary.json", summary), ("discrepancies.json", ledger), ("affected_observations.json", observations),
                          ("source_provenance.json", provenance), ("preservation.json", {"unchanged": True, "files_sha256": before})):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "discrepancies.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)
    print(json.dumps({"counts": counts, "repair_proposal": repair_counts, "protected_files": len(before)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
