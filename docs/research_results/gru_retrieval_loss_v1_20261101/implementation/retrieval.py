import numpy as np

def normalize(x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be finite matrices")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("Zero retrieval embedding")
    return x / norms

def ranks(queries, candidates, target_keys, candidate_keys, chunk_size=128):
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("Deduplicate candidate sentences before retrieval")
    if len(queries) != len(target_keys) or len(candidates) != len(candidate_keys) or not len(queries):
        raise ValueError("Invalid retrieval alignment")
    lookup = {key: i for i, key in enumerate(candidate_keys)}
    q, c = normalize(queries), normalize(candidates)
    out = []
    for start in range(0, len(q), chunk_size):
        scores = q[start:start+chunk_size] @ c.T
        for row, key in zip(scores, target_keys[start:start+chunk_size]):
            if key not in lookup:
                raise ValueError("Target absent from candidate pool")
            truth = row[lookup[key]]
            # Average tied rank avoids optimistic top-1 when all scores are equal.
            out.append(1 + int((row > truth).sum()) + .5 * (int((row == truth).sum()) - 1))
    return np.asarray(out, dtype=float)

def summarize(rank, ks=(1, 5, 10)):
    rank = np.asarray(rank)
    if not len(rank):
        raise ValueError("No retrieval observations")
    result = {f"recall@{k}": float((rank <= k).mean()) for k in ks}
    return dict(result, top1=float((rank <= 1).mean()), top5=float((rank <= 5).mean()),
                mrr=float((1 / rank).mean()), n_queries=len(rank))

def chance(n_candidates, ks=(1,5,10)):
    if n_candidates < 1:
        raise ValueError("Empty candidate pool")
    return {**{f"recall@{k}": min(k / n_candidates, 1.) for k in ks},
            "top1": 1/n_candidates, "top5": min(5/n_candidates, 1.),
            "mrr": float(np.sum(1 / np.arange(1,n_candidates+1)) / n_candidates)}
