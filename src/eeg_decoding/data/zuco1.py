"""Explicit sentenceData MATLAB adapter. No guessed sample rate or channel labels.

Schema JSON supplies subject/session/task/channels/sfreq, signal_field, text_field,
time_axis (0 or 1), units (uV or V), and optional person_id. Word-level band features
are not raw EEG and are deliberately not accepted as time series.
Level 5 files stream selected sentence fields; v7.3 files read HDF5 references.
"""
from pathlib import Path
import hashlib
import numpy as np
from .dataset import Record

def iter_mat(path, metadata, cache_dir, dataset="zuco1", *, skip_missing_eeg=False,
             skip_short_eeg=False, min_samples=2, audit=None):
    import h5py
    path, cache_dir = Path(path), Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = {"subject", "session", "task", "channels", "sfreq", "signal_field", "text_field", "time_axis", "units"}
    if not required <= metadata.keys():
        raise ValueError(f"MATLAB schema needs: {sorted(required)}")
    if metadata["units"] not in {"uV", "V"} or metadata["time_axis"] not in (0, 1):
        raise ValueError("Specify units and time_axis explicitly")
    if not isinstance(min_samples, int) or min_samples < 2:
        raise ValueError("min_samples must be an integer >=2")
    sf, tf = metadata["signal_field"], metadata["text_field"]
    if audit is None:
        audit = {}
    audit.update(source=str(path.resolve()), imported=0, skipped=[], index_base=0)
    def convert(i, signal, text, matlab_hdf=False):
        x = np.asarray(signal, dtype=np.float32)
        # ZuCo exports retain sentence slots with scalar NaN/empty EEG. Skip
        # only these explicit missing placeholders, never a channel/schema error.
        missing = x.size == 0 or (x.size == 1 and np.isnan(x).all())
        if missing:
            if not skip_missing_eeg:
                raise ValueError(f"Sentence {i}: missing EEG; use --skip-missing-eeg to exclude and audit this record")
            audit["skipped"].append({"sentence_id":str(i), "matlab_index":i+1, "reason":"missing_eeg"})
            return None
        # MATLAB HDF5 dimensions are reversed on disk.
        if matlab_hdf:
            x = x.T
        if metadata["time_axis"] == 0:
            x = x.T
        if x.ndim != 2 or x.shape[0] != len(metadata["channels"]):
            raise ValueError(f"Sentence {i}: expected raw C x T EEG, not band features; shape={x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"Sentence {i}: waveform contains NaN/Inf; inspect signal quality before importing")
        if x.shape[1] < min_samples:
            if not skip_short_eeg:
                raise ValueError(f"Sentence {i}: insufficient temporal samples ({x.shape[1]} < {min_samples})")
            audit["skipped"].append({"sentence_id":str(i), "matlab_index":i+1,
                                     "reason":"insufficient_temporal_samples",
                                     "n_samples":x.shape[1], "min_samples":min_samples})
            return None
        if metadata["units"] == "V":
            x *= 1e6
        digest = hashlib.sha256(f"{path.resolve()}:{path.stat().st_mtime_ns}:{i}".encode()).hexdigest()
        dest = cache_dir / f"{digest}.npz"
        np.savez_compressed(dest, eeg=x)
        audit["imported"] += 1
        return Record(dataset, metadata["subject"], metadata["session"], metadata["task"], str(i), str(text),
                      str(dest), metadata["channels"], float(metadata["sfreq"]),
                      provenance={"format": "MATLAB", "source": str(path.resolve()), "units": "uV", "matlab_sentence_index":i+1,
                                  **({"person_id": metadata["person_id"]} if "person_id" in metadata else {})})
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            group = f["sentenceData"]
            signals, texts = group[sf], group[tf]
            if signals.size != texts.size:
                raise ValueError("Sentence/EEG reference count mismatch")
            audit["total_sentences"] = signals.size
            for i in range(signals.size):
                idx = np.unravel_index(i, signals.shape)
                signal = f[signals[idx]][()] if signals[idx] else np.array([])
                text = ''.join(chr(int(c)) for c in f[texts[idx]][()].flatten(order="F"))
                record = convert(i, signal, text, matlab_hdf=True)
                if record is not None:
                    yield record
    else:
        from .mat5 import iter_sentence_fields
        for i, signal, text in iter_sentence_fields(path, sf, tf, audit=audit):
            record = convert(i, signal, text)
            if record is not None:
                yield record
