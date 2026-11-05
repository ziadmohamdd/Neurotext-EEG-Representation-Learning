"""Read-only CPU reconstruction of cached text targets from pinned local weights."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib
import json
from pathlib import Path
import unicodedata
import numpy as np
import torch


def main():
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    output=Path('results/pairing_audit_20261030/text_cache_audit.json')
    if output.exists(): raise FileExistsError(output)
    summary=json.loads(Path('data/processed/zuco1_batch/summary.json').read_text())
    path=Path(summary['text_cache']); before=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as f:
        keys=f['keys'].tolist(); old=f['embeddings'].astype(np.float32)
        metadata=json.loads(str(f['metadata'].item()))
    texts={}
    for line in Path('data/processed/zuco1_batch/combined.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        key=hashlib.sha256(' '.join(unicodedata.normalize('NFKC',r['text']).casefold().split()).encode()).hexdigest()
        texts[key]=r['text']
    assert sorted(texts)==keys and metadata['resolved_revision']
    model=SentenceTransformer(metadata['model'],revision=metadata['resolved_revision'],
                              device='cpu',local_files_only=True)
    new=model.encode([texts[k] for k in keys],batch_size=4,normalize_embeddings=True,
                     convert_to_numpy=True,show_progress_bar=False)
    old/=np.linalg.norm(old,axis=1,keepdims=True)
    cosine=np.sum(old*new,axis=1)
    nearest=(new @ old.T).argmax(1)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    report=dict(cache=str(path),sha256=before,n_sentences=len(keys),metadata=metadata,
                cpu_local_pinned_weights_only=True,original_cache_unchanged=True,
                minimum_same_key_cosine=float(cosine.min()),mean_same_key_cosine=float(cosine.mean()),
                max_absolute_embedding_difference=float(np.abs(old-new).max()),
                same_key_nearest_cached_embedding_count=int((nearest==np.arange(len(keys))).sum()),
                mismatched_nearest_keys=[keys[i] for i in range(len(keys)) if nearest[i]!=i],
                interpretation='Numerical reconstruction audit of key/vector correspondence, not a model training or decoding result')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
