"""Explicit sentenceData MATLAB adapter. No guessed sample rate or channel labels.

Schema JSON supplies subject/session/task/channels/sfreq, signal_field, text_field,
time_axis (0 or 1), units (uV or V), and optional person_id. Word-level band features
are not raw EEG and are deliberately not accepted as time series.
"""
from pathlib import Path
import hashlib
import numpy as np
from .dataset import Record

def iter_mat(path, metadata, cache_dir, dataset="zuco1", *, skip_missing_eeg=False, audit=None):
    import h5py
    from scipy.io import loadmat
    path, cache_dir = Path(path), Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = {"subject", "session", "task", "channels", "sfreq", "signal_field", "text_field", "time_axis", "units"}
    if not required <= metadata.keys():
        raise ValueError(f"MATLAB schema needs: {sorted(required)}")
    if metadata["units"] not in {"uV", "V"} or metadata["time_axis"] not in (0, 1):
        raise ValueError("Specify units and time_axis explicitly")
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
        if x.ndim != 2 or x.shape[0] != len(metadata["channels"]) or x.shape[1] < 2:
            raise ValueError(f"Sentence {i}: expected raw C x T EEG, not band features")
        if not np.isfinite(x).all():
            raise ValueError(f"Sentence {i}: waveform contains NaN/Inf; inspect signal quality before importing")
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
        if path.stat().st_size > 512 * 2**20:
            raise ValueError("MAT v5 >512 MiB: export sentence shards or MATLAB v7.3 for bounded-memory reading")
        # scipy cannot lazily decompress v5 variables; bound to one small subject file.
        mat = loadmat(path, simplify_cells=True, variable_names=["sentenceData"])
        rows = mat["sentenceData"]
        if isinstance(rows, dict):
            rows = [rows]
        audit["total_sentences"] = len(rows)
        for i, row in enumerate(rows):
            if sf not in row or tf not in row:
                raise ValueError(f"No {sf}/{tf}: supply correct raw sentence schema; do not concatenate word band features")
            record = convert(i, row[sf], row[tf])
            if record is not None:
                yield record
