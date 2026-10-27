def compatibility(source, target, require_identity_audit=True):
    if not source or not target:
        raise ValueError("Empty source or target")
    source_versions, target_versions = {r.dataset for r in source}, {r.dataset for r in target}
    if len(source_versions) != 1 or len(target_versions) != 1 or source_versions == target_versions:
        raise ValueError("Cross-dataset transfer requires distinct single-version source/target")
    if any(r.channels != source[0].channels or r.sfreq != source[0].sfreq for r in source + target):
        raise ValueError("Harmonize channel order, feature dimension and sampling rate before transfer")
    if {r.task for r in source} != {r.task for r in target}:
        raise ValueError("Task conditions differ; restrict both datasets to comparable conditions (e.g. nr)")
    if require_identity_audit and any("person_id" not in r.provenance for r in source + target):
        raise ValueError("Participant identity audit missing: add verified person_id to all records")
    overlap_subjects = {r.subject_key for r in source} & {r.subject_key for r in target}
    if overlap_subjects:
        raise ValueError(f"Cross-dataset participant leakage: {overlap_subjects}")
    overlap = {r.key for r in source} & {r.key for r in target}
    return {"shared_channels": source[0].channels, "sfreq": source[0].sfreq,
            "tasks": sorted({r.task for r in source}), "sentence_overlap": len(overlap),
            "source_observations": len(source), "target_observations": len(target),
            "protocol": "zero_shot_no_target_fitting", "identity_audited": require_identity_audit}

def transfer_split(source, target, seed=42, unseen_sentences=False):
    import numpy as np
    compatibility(source, target)
    subjects = sorted({r.subject_key for r in source})
    if len(subjects) < 2:
        raise ValueError("Need two source subjects for train/validation")
    np.random.default_rng(seed).shuffle(subjects)
    val = set(subjects[:max(1, len(subjects)//5)])
    splits = {"train": [r for r in source if r.subject_key not in val],
              "val": [r for r in source if r.subject_key in val], "test": target}
    if unseen_sentences:
        seen = {r.key for r in source}
        splits["test"] = [r for r in target if r.key not in seen]
    from .cross_subject import check_disjoint
    check_disjoint(splits)
    return splits
