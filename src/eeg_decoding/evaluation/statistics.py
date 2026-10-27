import numpy as np

def subject_bootstrap(values, subjects, seed=42, iterations=2000):
    """Cluster bootstrap of equal-subject macro means, not independent sentence trials."""
    values, subjects = np.asarray(values), np.asarray(subjects)
    unique = sorted(set(subjects))
    if len(unique) < 2:
        return {"mean": float(values.mean()), "ci95": None, "reason": "fewer than two subjects"}
    means = np.array([values[subjects == s].mean() for s in unique])
    rng = np.random.default_rng(seed)
    boots = means[rng.integers(0, len(means), (iterations, len(means)))].mean(1)
    return {"mean": float(means.mean()), "ci95": np.quantile(boots, [.025,.975]).tolist(),
            "n_subjects": len(unique), "iterations": iterations}
