from collections import Counter
import hashlib
import json
from .dataset import load_signal

def manifest_digest(records):
    # Include source provenance and array bytes: stale caches cannot silently reuse changed data.
    h = hashlib.sha256()
    for r in records:
        from dataclasses import asdict
        metadata = asdict(r)
        metadata.pop("signal_path")
        # Physical paths may change between Windows, Drive and Colab staging.
        # Identity, preprocessing, quality and actual cached bytes remain binding.
        metadata["provenance"] = {k:v for k,v in metadata["provenance"].items()
                                  if k not in {"source", "previous_source", "alignment"}}
        h.update(json.dumps(metadata, sort_keys=True).encode())
        with open(r.signal_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()

def validate_records(records, inspect_signals=True):
    errors, seen, counts, sentences = [], set(), Counter(), Counter()
    for r in records:
        identity = (r.dataset, r.subject, r.session, r.task, r.sentence_id)
        if identity in seen:
            errors.append(f"Repeated observation: {identity}")
        seen.add(identity)
        counts[r.subject_key] += 1
        sentences[r.key] += 1
        if r.sfreq <= 0 or len(r.channels) != len(set(r.channels)) or not r.channels:
            errors.append(f"Invalid channels/sfreq: {identity}")
        if inspect_signals:
            try:
                load_signal(r)
            except (ValueError, OSError) as exc:
                errors.append(f"{identity}: {exc}")
    return {"n_observations": len(records), "subjects": dict(counts), "sentence_counts": dict(sentences),
            "n_unique_sentences": len(sentences), "errors": errors, "valid": bool(records) and not errors}
