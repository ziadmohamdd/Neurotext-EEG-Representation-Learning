import numpy as np
from ..data.validation import manifest_digest

def check_disjoint(splits, sentences=False):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if {r.subject_key for r in splits[a]} & {r.subject_key for r in splits[b]}:
                raise ValueError(f"Subject leakage between {a} and {b}")
            if sentences and {r.key for r in splits[a]} & {r.key for r in splits[b]}:
                raise ValueError(f"Sentence leakage between {a} and {b}")
    if any(not rows for rows in splits.values()):
        raise ValueError("Empty split; add subjects/sentences or change allocation")

def group_split(records, seed=42, held_out_sentences=False):
    subjects = sorted({r.subject_key for r in records})
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = max(1, int(len(subjects) * .2))
    assigned = {"test": set(subjects[:n]), "val": set(subjects[n:2*n]), "train": set(subjects[2*n:])}
    splits = {name: [r for r in records if r.subject_key in group] for name, group in assigned.items()}
    if held_out_sentences:
        keys = sorted({r.key for r in records})
        if len(keys) < 3:
            raise ValueError("Need at least three unique sentences")
        rng.shuffle(keys)
        k = max(1, int(len(keys) * .2))
        content = {"test": set(keys[:k]), "val": set(keys[k:2*k]), "train": set(keys[2*k:])}
        splits = {name: [r for r in rows if r.key in content[name]] for name, rows in splits.items()}
    check_disjoint(splits, sentences=held_out_sentences)
    return splits

def split_manifest(splits, seed, protocol):
    return {"schema_version": 1, "seed": seed, "protocol": protocol,
            "splits": {name: {"subjects": sorted({r.subject_key for r in rows}),
                              "observations": [[r.dataset, r.subject, r.session, r.task, r.sentence_id] for r in rows],
                              "sentence_keys": sorted({r.key for r in rows}), "digest": manifest_digest(rows)}
                       for name, rows in splits.items()}}

def loso(records):
    for subject in sorted({r.subject_key for r in records}):
        rest = [r for r in records if r.subject_key != subject]
        val_subject = sorted({r.subject_key for r in rest})[-1]
        split = {"train": [r for r in rest if r.subject_key != val_subject],
                 "val": [r for r in rest if r.subject_key == val_subject],
                 "test": [r for r in records if r.subject_key == subject]}
        check_disjoint(split)
        yield subject, split
