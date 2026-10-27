"""Stage selected cache shards from persistent storage without redownloading raw EEG."""
from dataclasses import replace
from pathlib import Path
import hashlib
import shutil
from ..data.dataset import read_manifest, write_manifest

def checksum(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def stage_manifest(manifest, destination):
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    staged=[]
    for record in read_manifest(manifest):
        source=Path(record.signal_path)
        sha=checksum(source)
        target=destination/(sha+".npz")
        if not target.exists() or checksum(target)!=sha:
            temp=target.with_suffix(".partial")
            shutil.copyfile(source,temp)
            if checksum(temp)!=sha:
                raise OSError("Staged cache checksum mismatch")
            temp.replace(target)
        staged.append(replace(record,signal_path=str(target)))
    output=destination/"manifest.jsonl"
    write_manifest(output,staged)
    return output
