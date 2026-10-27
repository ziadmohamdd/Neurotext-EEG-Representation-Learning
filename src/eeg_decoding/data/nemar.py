"""Raw BIDS is not sentence-aligned MATLAB. Explicit alignment TSV is mandatory.

Columns: recording (relative _eeg.vhdr path), onset, duration, sentence_id, text,
subject, session, task. Times in seconds on the EEG sample clock, not eye tracker clock.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
from .dataset import Record
from ..preprocessing.segmentation import sample_bounds

REQUIRED = {"recording", "onset", "duration", "sentence_id", "text", "subject", "session", "task"}

def iter_bids(root, alignment_tsv, dataset, cache_dir):
    import mne
    root, cache_dir = Path(root).resolve(), Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(alignment_tsv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError(f"Verified sentence alignment TSV needs columns: {sorted(REQUIRED)}")
        rows = sorted(reader, key=lambda r: r["recording"])
    current, raw = None, None
    try:
        for row in rows:
            # Permit git-annex symlinks to an external object store, but never a
            # TSV path outside the explicitly selected dataset tree.
            path = Path(os.path.abspath(root / row["recording"]))
            if not path.is_relative_to(root):
                raise ValueError("Recording escapes dataset root")
            if path != current:
                if raw is not None:
                    raw.close()
                if not path.is_file():
                    raise FileNotFoundError(f"Retrieve git-annex recording first: {path}")
                raw = mne.io.read_raw(path, preload=False, verbose="ERROR")
                channels_path = path.with_name(path.name.replace("_eeg.vhdr","_channels.tsv").replace("_eeg.set","_channels.tsv"))
                if channels_path != path and channels_path.is_file():
                    with channels_path.open(encoding="utf-8-sig", newline="") as channel_file:
                        channel_rows = list(csv.DictReader(channel_file,delimiter="\t"))
                    names = [r["name"] for r in channel_rows]
                    if names != raw.ch_names:
                        raise ValueError("BIDS channel TSV and recording header channel order differ")
                    raw.info["bads"] = [r["name"] for r in channel_rows if r.get("status") == "bad"]
                    types = {r["name"]:r["type"].lower() for r in channel_rows
                             if r.get("type","").lower() in {"eeg","eog","ecg","emg","misc","stim"}}
                    raw.set_channel_types(types,verbose="ERROR")
                json_path = path.with_suffix(".json")
                if json_path.is_file():
                    sidecar = json.loads(json_path.read_text())
                    if "SamplingFrequency" in sidecar and sidecar["SamplingFrequency"] != raw.info["sfreq"]:
                        raise ValueError("BIDS sidecar/header sample rates differ")
                current = path
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            if not len(picks):
                raise ValueError("No valid EEG channels in recording")
            start, stop = sample_bounds(float(row["onset"]), float(row["duration"]), raw.info["sfreq"], raw.n_times)
            # MNE returns volts; internal raw signals are always microvolts.
            x = (raw.get_data(picks=picks, start=start, stop=stop) * 1e6).astype(np.float32)
            digest = hashlib.sha256((str(path) + str(path.stat().st_mtime_ns) + str(start) + ':' + str(stop)).encode()).hexdigest()
            target = cache_dir / f"{digest}.npz"
            np.savez_compressed(target, eeg=x)
            yield Record(dataset, row["subject"], row["session"], row["task"], row["sentence_id"],
                         row["text"], str(target), [raw.ch_names[p] for p in picks], float(raw.info["sfreq"]),
                         provenance={"format": "BIDS", "source": str(path), "onset": float(row["onset"]),
                                     "duration": float(row["duration"]), "units": "uV", "alignment": str(Path(alignment_tsv).resolve()),
                                     **({"person_id":row["person_id"]} if row.get("person_id") else {})})
    finally:
        if raw is not None:
            raw.close()
