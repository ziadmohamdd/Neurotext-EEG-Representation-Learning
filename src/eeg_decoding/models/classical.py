import numpy as np
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

def eeg_features(x, sfreq):
    frequencies, psd = welch(x, fs=sfreq, nperseg=min(x.shape[-1], int(sfreq * 2)), axis=-1)
    bands = []
    for low, high in BANDS:
        selected = (frequencies >= low) & (frequencies < high)
        bands.append(np.log1p(np.trapezoid(psd[:, selected], frequencies[selected], axis=-1)))
    stats = [x.mean(-1), x.std(-1), np.ptp(x, axis=-1), np.sqrt((x*x).mean(-1)),
             np.abs(np.diff(x, axis=-1)).mean(-1)]
    return np.concatenate(stats + bands).astype(np.float32)

def _float64(features):
    # Keep disk-backed feature caches compact; solve the small regression problem
    # in double precision, including standardization, for correlated EEG features.
    return np.asarray(features, dtype=np.float64)

def ridge_model(alpha=1.):
    return make_pipeline(FunctionTransformer(_float64, validate=False),
                         StandardScaler(), Ridge(alpha=alpha, solver="svd"))


def ridge_path(features, targets, alphas):
    """Fit a regularization path with one train-only scaling and SVD.

    Matches sklearn's centered, float64 SVD ridge, including its singular-value
    cutoff. Yield independent coefficient arrays in the existing artifact format.
    """
    from scipy.linalg import svd
    alphas = [float(a) for a in alphas]
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("ridge_alphas must contain finite positive values")
    scaler = StandardScaler().fit(_float64(features))
    x = scaler.transform(_float64(features))
    y = _float64(targets)
    x_mean, y_mean = x.mean(0), y.mean(0)
    u, s, vt = svd(x - x_mean, full_matrices=False)
    keep = s > 1e-15
    uty = u[:, keep].T @ (y - y_mean)
    for alpha in alphas:
        coef = (vt[keep].T @ ((s[keep] / (s[keep]**2 + alpha))[:, None] * uty)).T
        yield alpha, dict(mean=scaler.mean_, scale=scaler.scale_, coef=coef,
                          intercept=y_mean - x_mean @ coef.T)
