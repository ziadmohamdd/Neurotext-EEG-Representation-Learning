"""ZuCo 2.0 MATLAB entry point: schemas must be inspected independently."""
from .zuco1 import iter_mat as _iter_mat

def iter_mat(path, metadata, cache_dir, *, skip_missing_eeg=False,
             skip_short_eeg=False, min_samples=2, audit=None):
    return _iter_mat(path, metadata, cache_dir, dataset="zuco2",
                     skip_missing_eeg=skip_missing_eeg, skip_short_eeg=skip_short_eeg,
                     min_samples=min_samples, audit=audit)
