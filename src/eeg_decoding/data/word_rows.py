"""Read word-locked EEG rows from ZuCo source files. Unwanted sentences' word fields are skipped, never kept.

Only TRT band amplitudes, TRT, nFixations and the first rawEEG cell are decoded. Word
text, eye-tracking traces and word boxes are never read here; tokens come from the manifest.
"""
import math
import h5py
import numpy as np
from . import mat5
from .trt_view import BANDS, field_names

MEASURES = {'TRT', 'nFixations', *('TRT_' + b for b in BANDS)}
BINS, BIN_SAMPLES = 6, 25
MINIMUM_BIN_SAMPLES = 5


def first_cell(element, endian):
    """The first fixation's snippet as (105, N), or None when it is missing or a NaN sentinel."""
    if element.size == 0:
        return None
    flags, shape, _ = mat5._header(element.payload, endian)
    if flags & 255 != 1:
        return None
    first = None
    for k in range(math.prod(shape)):
        child = mat5._Element(element.payload, endian)
        if k == 0:
            first = mat5._selected(child, endian, False, mat5.MAX_FIELD_BYTES)
        child.finish()
    return first


def _level5_words(element, endian):
    if not element.size:
        return []
    flags, shape, _ = mat5._header(element.payload, endian)
    if flags & 255 != 2:
        return []
    fields = field_names(element.payload, endian)
    words = []
    for _ in range(math.prod(shape)):
        values, raw = {}, None
        for name in fields:
            child = mat5._Element(element.payload, endian)
            if name in MEASURES:
                values[name] = mat5._selected(child, endian, False, mat5.MAX_FIELD_BYTES)
            elif name == 'rawEEG':
                raw = first_cell(child, endian)
            child.finish()
        words.append((values, raw))
    return words


def level5_sentences(path, wanted):
    with open(path, 'rb') as f:
        header = mat5._exact(f, 128)
        endian = {b'IM': '<', b'MI': '>'}.get(header[126:128])
        if endian is None:
            raise ValueError('Expected a Level-5 file')
        outer = mat5._Element(f, endian)
        compressed = mat5._Inflater(outer.payload) if outer.kind == mat5.COMPRESSED else None
        matrix = mat5._Element(compressed, endian) if compressed else outer
        flags, shape, name = mat5._header(matrix.payload, endian)
        if name != 'sentenceData' or flags & 255 != 2:
            raise ValueError('Expected sentenceData struct')
        fields = field_names(matrix.payload, endian)
        for index in range(math.prod(shape)):
            words = None
            for name in fields:
                child = mat5._Element(matrix.payload, endian)
                if name == 'word' and index in wanted:
                    words = _level5_words(child, endian)
                child.finish()
            if index in wanted:
                yield index, words or []


def _hdf_value(f, ref):
    node = f[ref]
    if not isinstance(node, h5py.Dataset) or node.attrs.get('MATLAB_empty', 0):
        return np.empty((0, 0))
    return node[()]


def _hdf_first_cell(f, ref):
    node = f[ref]
    if not isinstance(node, h5py.Dataset) or node.attrs.get('MATLAB_empty', 0) or h5py.check_dtype(ref=node.dtype) is None:
        return None
    cells = node[()].flatten(order='F')
    if not len(cells):
        return None
    value = _hdf_value(f, cells[0])
    # HDF5 stores MATLAB's (105, N) as (N, 105).
    return value.transpose() if value.ndim == 2 else value


def hdf_sentences(path, wanted):
    with h5py.File(path, 'r') as f:
        references = f['sentenceData']['word'][()].flatten(order='F')
        for index in sorted(wanted):
            if index >= len(references):
                raise ValueError(f'Sentence {index} is not in {path}')
            node = f[references[index]]
            if not isinstance(node, h5py.Group):
                yield index, []
                continue
            columns = {name: node[name][()].flatten(order='F') for name in node if name in MEASURES or name == 'rawEEG'}
            count = len(next(iter(columns.values()))) if columns else 0
            words = []
            for i in range(count):
                values = {name: _hdf_value(f, column[i]) for name, column in columns.items() if name != 'rawEEG'}
                words.append((values, _hdf_first_cell(f, columns['rawEEG'][i]) if 'rawEEG' in columns else None))
            yield index, words


def source_sentences(path, wanted):
    return hdf_sentences(path, wanted) if h5py.is_hdf5(path) else level5_sentences(path, wanted)


def band_vector(values, keep):
    """Log band amplitudes without the dropped channel, or None unless every value is finite."""
    bands = []
    for band in BANDS:
        a = np.asarray(values.get('TRT_' + band, np.empty(0)), dtype=np.float64)
        if a.size != 105:
            return None
        bands.append(a.reshape(-1)[keep])
    vector = np.concatenate(bands)
    if not np.isfinite(vector).all():
        return None
    if (vector < 0).any():
        raise ValueError('Negative source band amplitude')
    return np.log1p(vector).astype(np.float32)


def scalar(values, name):
    a = np.asarray(values.get(name, np.empty(0)), dtype=np.float64)
    return float(a.flat[0]) if a.size == 1 else np.nan


def waveform_bins(raw, keep):
    """Mean amplitude in 50 ms bins up to 300 ms at 500 Hz; NaN where the fixation has ended."""
    output = np.full((BINS, len(keep)), np.nan, dtype=np.float32)
    if raw is None or np.ndim(raw) != 2 or raw.shape[0] != 105 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        return output.reshape(-1)
    for b in range(BINS):
        part = raw[keep, b * BIN_SAMPLES:(b + 1) * BIN_SAMPLES]
        if part.shape[1] >= MINIMUM_BIN_SAMPLES:
            output[b] = part.mean(1)
    return output.reshape(-1)


def extract_rows(path, wanted, keep):
    """Rows for one reader's file. `wanted` maps a zero-based sentence index to its token count."""
    sentence, slot, bands, waves, durations, fixations = [], [], [], [], [], []
    seen = set()
    for index, words in source_sentences(path, set(wanted)):
        seen.add(index)
        if words and len(words) != wanted[index]:
            raise ValueError(f'{path}: sentence {index} has {len(words)} word records and {wanted[index]} tokens')
        for position, (values, raw) in enumerate(words):
            vector = band_vector(values, keep)
            if vector is None:
                continue
            sentence.append(index)
            slot.append(position)
            bands.append(vector)
            waves.append(waveform_bins(raw, keep))
            durations.append(scalar(values, 'TRT'))
            fixations.append(scalar(values, 'nFixations'))
    if seen != set(wanted):
        raise ValueError(f'{path}: missing sentences {sorted(set(wanted) - seen)[:5]}')
    width = 8 * len(keep)
    return {'sentence': np.array(sentence, dtype=np.int32), 'slot': np.array(slot, dtype=np.int32),
            'bands': np.array(bands, dtype=np.float32).reshape(-1, width), 'waves': np.array(waves, dtype=np.float32).reshape(-1, BINS * len(keep)),
            'durations': np.array(durations, dtype=np.float32), 'fixations': np.array(fixations, dtype=np.float32)}


__all__ = ['BINS', 'band_vector', 'extract_rows', 'source_sentences', 'waveform_bins']
