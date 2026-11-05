"""Opt-in exported word targets, separate from the EEG-only TRT reader.

Read only original training-reader slots. Text is target metadata, never input.
"""
import math
import numpy as np
from . import mat5
from .trt_view import BANDS, field_names


def labelled_words(element, endian):
    flags, shape, _ = mat5._header(element.payload, endian)
    if flags & 255 != 2:
        raise ValueError('Expected an exported word struct')
    fields = field_names(element.payload, endian)
    if 'content' not in fields:
        raise ValueError('Missing exported word content')
    labels, vectors = [], []
    for _ in range(math.prod(shape)):
        values = {}
        for name in fields:
            child = mat5._Element(element.payload, endian)
            if name == 'content' or name in {'TRT_' + b for b in BANDS}:
                values[name] = mat5._selected(child, endian, name == 'content', mat5.MAX_FIELD_BYTES)
            child.finish()  # raw EEG/ET, durations, positions and all other fields skipped.
        label = values['content']
        if not isinstance(label, str) or not label:
            raise ValueError('Empty or invalid exported word label')
        labels.append(label)
        bands = []
        for b in BANDS:
            value = values.get('TRT_' + b, np.empty(0))
            if not value.size:
                bands.append(np.full(105, np.nan))
            elif value.shape in ((1, 105), (105, 1)):
                bands.append(value.reshape(-1))
            else:
                raise ValueError('Unexpected exported word EEG shape')
        vectors.append(np.concatenate(bands))
    return labels, np.asarray(vectors).reshape(-1, 840)


def iter_labelled_training_slots(path, wanted):
    """Yield only explicitly requested slots, retaining exact word/parent strings."""
    wanted = set(wanted)
    with open(path, 'rb') as f:
        header = mat5._exact(f, 128)
        endian = {b'IM': '<', b'MI': '>'}.get(header[126:128])
        if endian is None:
            raise ValueError('Requires audited Level-5 source')
        outer = mat5._Element(f, endian)
        compressed = mat5._Inflater(outer.payload) if outer.kind == mat5.COMPRESSED else None
        matrix = mat5._Element(compressed, endian) if compressed else outer
        flags, shape, name = mat5._header(matrix.payload, endian)
        if name != 'sentenceData' or flags & 255 != 2:
            raise ValueError('Expected sentenceData struct')
        fields = field_names(matrix.payload, endian)
        found = set()
        for slot in range(math.prod(shape)):
            parent = None; pair = None
            for field in fields:
                child = mat5._Element(matrix.payload, endian)
                if slot in wanted:
                    if field == 'content':
                        parent = mat5._selected(child, endian, True, mat5.MAX_FIELD_BYTES)
                    elif field == 'word':
                        pair = labelled_words(child, endian)
                child.finish()
            if slot in wanted:
                if parent is None or pair is None:
                    raise ValueError('Requested source slot lacks parent or word data')
                labels, words = pair
                if ' '.join(labels) != parent:
                    raise ValueError('Exported words do not exactly reconstruct parent')
                found.add(slot)
                yield slot, parent, labels, words
        if found != wanted or matrix.payload.remaining:
            raise ValueError('Requested slots missing or trailing struct bytes')
        matrix.finish()
        if compressed:
            compressed.finish(); outer.finish()


def check_pairing(parent, labels, source_words, expected_parent, cached_words):
    if parent != expected_parent or ' '.join(labels) != expected_parent:
        raise ValueError('Unchanged exported target mismatch')
    if len(labels) != len(cached_words) or source_words.shape != cached_words.shape:
        raise ValueError('Word-to-cache row mismatch')
    if not np.array_equal(source_words, cached_words, equal_nan=True):
        raise ValueError('Source TRT values differ from saved word cache')
