"""Training-only word projection and ordered pooling for a bounded TRT pilot."""
import numpy as np
from sklearn.decomposition import PCA


def fit_projector(sequences, components=16, max_words=8192, seed=42):
    words = np.concatenate([x[np.isfinite(x).all(1)] for x in sequences])
    if len(words) < components:
        raise ValueError("Too few finite training words")
    center = words.mean(0)
    scale = words.std(0)
    scale[scale < 1e-12] = 1.0
    rng = np.random.default_rng(seed)
    sample = words[rng.choice(len(words), min(max_words, len(words)), replace=False)]
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit((sample - center) / scale)
    return {"center": center, "scale": scale, "pca_mean": pca.mean_,
            "components": pca.components_, "explained_variance_ratio": pca.explained_variance_ratio_,
            "training_words": len(words), "pca_fit_words": len(sample)}


def pooled(words, projector, segments=4, permutation_seed=None):
    if words.ndim != 2 or words.shape[1] != len(projector["center"]):
        raise ValueError("Unexpected word feature width")
    valid = np.isfinite(words).all(1)
    if not np.all(valid | ~np.isfinite(words).any(1)):
        raise ValueError("Partial word missingness needs a new protocol")
    finite = words[valid]
    if permutation_seed is not None:
        finite = finite[np.random.default_rng(permutation_seed).permutation(len(finite))]
    z = np.zeros((len(words), len(projector["components"])))
    z[valid] = (((finite - projector["center"]) / projector["scale"] - projector["pca_mean"])
                @ projector["components"].T)
    values, fractions = [], []
    for indices in np.array_split(np.arange(len(words)), segments):
        selected = indices[valid[indices]]
        values.append(z[selected].mean(0) if len(selected) else np.zeros(z.shape[1]))
        fractions.append(len(selected) / len(indices) if len(indices) else 0.0)
    return np.r_[np.concatenate(values), fractions]


def missingness(words, segments=4):
    valid = np.isfinite(words).all(1)
    fractions = [valid[ix].mean() if len(ix) else 0.0
                 for ix in np.array_split(np.arange(len(words)), segments)]
    return np.r_[fractions, len(words), valid.mean() if len(words) else 0.0]


def select_alpha(rows):
    """Each row contains equal-reader mean inner-training Recall@1."""
    if not rows:
        raise ValueError("No inner folds")
    return max(rows, key=lambda r: (r["macro_reader_recall@1"], r["alpha"]))["alpha"]
