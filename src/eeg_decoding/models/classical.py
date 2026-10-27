import numpy as np
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

def ridge_model(alpha=1.):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
