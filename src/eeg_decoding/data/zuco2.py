"""ZuCo 2.0 MATLAB entry point: schemas must be inspected independently."""
from .zuco1 import iter_mat as _iter_mat

def iter_mat(path, metadata, cache_dir):
    return _iter_mat(path, metadata, cache_dir, dataset="zuco2")
