"""Bounded-memory MATLAB Level 5 sentence-struct reader.

Only selected raw numeric matrices and sentence character arrays are materialized.
Unused feature/word/cell fields are discarded in 64 KiB chunks, including inside
miCOMPRESSED variables. This is a narrow decoder, not a general loadmat replacement.
"""
import math
import struct
import zlib

import numpy as np

CHUNK = 64 * 1024
MAX_FIELD_BYTES = 64 * 2**20
MAX_METADATA_BYTES = 1024 * 1024
MATRIX, COMPRESSED = 14, 15
DTYPES = {1:'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4',
          7:'f4', 9:'f8', 12:'i8', 13:'u8'}


def _exact(stream, size):
    parts, remaining = [], size
    while remaining:
        chunk = stream.read(min(remaining, CHUNK))
        if not chunk:
            raise ValueError('Truncated MATLAB data element')
        parts.append(chunk)
        remaining -= len(chunk)
    return b''.join(parts)


def _discard(stream, size):
    while size:
        chunk = stream.read(min(size, CHUNK))
        if not chunk:
            raise ValueError('Truncated MATLAB data while skipping unused fields')
        size -= len(chunk)


class _Limited:
    def __init__(self, source, size):
        self.source, self.remaining = source, size

    def read(self, size):
        if size < 0:
            raise ValueError('Unbounded reads are forbidden')
        data = self.source.read(min(size, self.remaining))
        self.remaining -= len(data)
        return data

    def discard(self):
        _discard(self, self.remaining)


class _Inflater:
    """Incremental decompression with capped output and zlib checksum verification."""
    def __init__(self, compressed):
        self.source = compressed
        self.decoder = zlib.decompressobj()
        self.pending = b''

    def read(self, size):
        if size <= 0:
            return b''
        if self.decoder.eof:
            return b''
        while True:
            data = self.pending or self.source.read(CHUNK)
            try:
                output = self.decoder.decompress(data, min(size, CHUNK))
            except zlib.error as exc:
                raise ValueError('MATLAB compressed stream failed integrity verification') from exc
            self.pending = self.decoder.unconsumed_tail
            if self.decoder.eof and (self.decoder.unused_data or self.pending or self.source.remaining):
                raise ValueError('Unexpected trailing bytes inside MATLAB compressed element')
            if output:
                return output
            if self.decoder.eof:
                return b''
            if not data and not self.pending:
                raise ValueError('Truncated MATLAB compressed stream')

    def finish(self):
        # There must be no unparsed decompressed element after the declared matrix.
        if self.read(CHUNK):
            raise ValueError('Unexpected trailing decompressed MATLAB data')
        if not self.decoder.eof:
            raise ValueError('Incomplete MATLAB compressed stream')


class _Element:
    def __init__(self, stream, endian, allow_eof=False):
        first = stream.read(8)
        if not first:
            if allow_eof:
                self.kind = None
                return
            raise ValueError('Missing MATLAB data element')
        if len(first) < 8:
            first += _exact(stream, 8-len(first))
        word, size = struct.unpack(endian+'II', first)
        small_size = word >> 16
        self.stream = stream
        if small_size:
            self.kind, self.size = word & 0xffff, small_size
            if not 1 <= small_size <= 4:
                raise ValueError('Invalid small MATLAB element')
            import io
            self.payload = _Limited(io.BytesIO(first[4:4+small_size]), small_size)
            self.padding = 0
        else:
            self.kind, self.size = word, size
            self.payload = _Limited(stream, size)
            # MATLAB/SciPy compressed top-level elements are not 8-byte padded.
            self.padding = 0 if word == COMPRESSED else (-size) % 8

    def read_bytes(self, limit):
        if self.size > limit:
            raise ValueError(f'MATLAB field exceeds {limit/2**20:g} MiB materialization limit; export smaller sentence shards')
        return _exact(self.payload, self.size)

    def finish(self):
        self.payload.discard()
        _discard(self.stream, self.padding)


def _small_field(stream, endian, limit=MAX_METADATA_BYTES):
    element = _Element(stream, endian)
    data = element.read_bytes(limit)
    element.finish()
    return element.kind, data


def _header(matrix, endian):
    kind, flags = _small_field(matrix, endian, 8)
    if kind != 6 or len(flags) != 8:
        raise ValueError('Invalid MATLAB array flags')
    flags = struct.unpack(endian+'II', flags)[0]
    kind, dims = _small_field(matrix, endian, 32)
    if kind != 5 or len(dims) % 4:
        raise ValueError('Invalid MATLAB dimensions')
    shape = struct.unpack(endian+'i'*(len(dims)//4), dims)
    if len(shape) < 2 or any(d < 0 for d in shape):
        raise ValueError('Invalid MATLAB array shape')
    kind, name = _small_field(matrix, endian)
    if kind not in {1,2,16}:
        raise ValueError('Invalid MATLAB variable name encoding')
    return flags, shape, name.decode('utf-8')


def _selected(element, endian, text, max_field_bytes):
    if element.kind != MATRIX:
        raise ValueError('Selected sentence field must be a MATLAB matrix')
    if element.size == 0:
        return '' if text else np.empty((0,0), np.float32)
    flags, shape, _ = _header(element.payload, endian)
    cls, count = flags & 0xff, math.prod(shape)
    if flags & 0x800:
        raise ValueError('Complex EEG/text fields are unsupported')
    if not count:
        return '' if text else np.empty(shape, np.float32)
    if text:
        if cls != 4:
            raise ValueError('Sentence text field must be a MATLAB character array')
        data_type, data = _small_field(element.payload, endian, min(max_field_bytes, MAX_METADATA_BYTES))
        codecs = {1:'latin-1', 2:'latin-1', 4:'utf-16-le' if endian=='<' else 'utf-16-be',
                  16:'utf-8', 17:'utf-16-le' if endian=='<' else 'utf-16-be',
                  18:'utf-32-le' if endian=='<' else 'utf-32-be'}
        if data_type not in codecs:
            raise ValueError('Unsupported MATLAB character encoding')
        value = data.decode(codecs[data_type])
        if element.payload.remaining:
            raise ValueError('Unexpected extra data in sentence character field')
        return value.rstrip('\x00')
    if cls not in {6,7,8,9,10,11,12,13,14,15}:
        raise ValueError('EEG field must be a real numeric matrix, not word/cell features')
    data_type, data = _small_field(element.payload, endian, max_field_bytes)
    if data_type not in DTYPES:
        raise ValueError('Unsupported MATLAB numeric encoding')
    dtype = np.dtype(endian+DTYPES[data_type])
    if len(data) != count*dtype.itemsize or element.payload.remaining:
        raise ValueError('MATLAB numeric field dimensions/data length mismatch')
    # Own writable storage; the loader may convert volts to microvolts in-place.
    return np.frombuffer(data, dtype=dtype).reshape(shape, order='F').copy(order='F')


def _field_names(matrix, endian, signal_field, text_field):
    kind, data = _small_field(matrix, endian, 4)
    if kind != 5 or len(data) != 4:
        raise ValueError('Invalid MATLAB struct field-name length')
    width = struct.unpack(endian+'i', data)[0]
    kind, names = _small_field(matrix, endian)
    if kind not in {1,2} or width <= 0 or len(names) % width:
        raise ValueError('Invalid MATLAB struct field names')
    fields = [names[i:i+width].split(b'\0',1)[0].decode('utf-8') for i in range(0,len(names),width)]
    if len(set(fields)) != len(fields) or not {signal_field,text_field} <= set(fields):
        raise ValueError(f'Missing/duplicate {signal_field}/{text_field} in sentenceData')
    return fields


def _row(matrix, fields, endian, signal_field, text_field, max_field_bytes):
    selected = {}
    for field in fields:
        element = _Element(matrix, endian)
        if field in {signal_field,text_field}:
            selected[field] = _selected(element,endian,field==text_field,max_field_bytes)
        element.finish()
    return selected[signal_field], selected[text_field]


def iter_sentence_fields(path, signal_field='rawData', text_field='content', audit=None,
                         max_field_bytes=MAX_FIELD_BYTES):
    """Yield (original zero-based index, numeric signal, text) without loading a subject.

    Bounds are per selected field, independent of total compressed subject size.
    Other nested fields are discarded bytewise and never recursively decoded.
    """
    if signal_field == text_field:
        raise ValueError('Signal and text fields must be distinct')
    with open(path, 'rb') as source:
        header = _exact(source,128)
        endian = {b'IM':'<', b'MI':'>'}.get(header[126:128])
        if endian is None or struct.unpack(endian+'H',header[124:126])[0] != 0x100:
            raise ValueError('Expected MATLAB Level 5 header (use HDF5 reader for v7.3)')
        while True:
            outer = _Element(source,endian,allow_eof=True)
            if outer.kind is None:
                raise ValueError('No sentenceData struct found in MATLAB file')
            compressed = _Inflater(outer.payload) if outer.kind == COMPRESSED else None
            matrix = _Element(compressed,endian) if compressed else outer
            if matrix.kind != MATRIX:
                raise ValueError('Expected top-level MATLAB matrix')
            flags, shape, name = _header(matrix.payload,endian)
            if name != 'sentenceData':
                matrix.finish()
                if compressed:
                    compressed.finish()
                    outer.finish()
                continue
            cls = flags & 0xff
            if cls not in {1,2}:
                raise ValueError('sentenceData must be a MATLAB struct array or cell array of scalar structs')
            fields = _field_names(matrix.payload,endian,signal_field,text_field) if cls == 2 else None
            total = math.prod(shape)
            if audit is not None:
                audit.update(total_sentences=total, reader='mat5_incremental', max_selected_field_bytes=max_field_bytes)
            for index in range(total):
                if cls == 2:
                    signal, text = _row(matrix.payload,fields,endian,signal_field,text_field,max_field_bytes)
                else:
                    cell = _Element(matrix.payload,endian)
                    if cell.kind != MATRIX:
                        raise ValueError('Sentence cell must contain one scalar struct')
                    cell_flags, cell_shape, _ = _header(cell.payload,endian)
                    if cell_flags & 0xff != 2 or math.prod(cell_shape) != 1:
                        raise ValueError('Sentence cell must contain one scalar struct')
                    cell_fields = _field_names(cell.payload,endian,signal_field,text_field)
                    signal, text = _row(cell.payload,cell_fields,endian,signal_field,text_field,max_field_bytes)
                    if cell.payload.remaining:
                        raise ValueError('Unexpected trailing data in sentence cell')
                    cell.finish()
                yield index, signal, text
            if matrix.payload.remaining:
                raise ValueError('Unexpected trailing MATLAB struct data')
            matrix.finish()
            if compressed:
                compressed.finish()
                outer.finish()
            return
