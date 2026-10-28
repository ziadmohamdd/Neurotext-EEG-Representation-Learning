"""Read ordered EEGLAB channel metadata; HDF5 waveforms are never preloaded."""
from pathlib import Path
import hashlib
import h5py
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _deref(f, obj):
    if isinstance(obj, h5py.Dataset) and h5py.check_dtype(ref=obj.dtype) is not None:
        refs = obj[()].flatten(order='F')
        if refs.size == 1 and refs[0]:
            return f[refs[0]]
    return obj


def _text(f, obj):
    if h5py.check_dtype(ref=obj.dtype) is not None:
        return [_text(f, f[ref]) for ref in obj[()].flatten(order='F') if ref]
    values = obj[()].flatten(order='F')
    return ''.join(chr(int(c)) for c in values if int(c))


def _coordinate(value):
    values = np.asarray(value).ravel()
    if values.size != 1:
        return None
    number = float(values[0])
    return number if np.isfinite(number) else None


def read_eeg_metadata(path, include_coordinates=False):
    path = Path(path)
    coordinates = None
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            eeg = _deref(f, f['EEG'])
            locations = _deref(f, eeg['chanlocs'])
            labels = _text(f, locations['labels'])
            if isinstance(labels, str):
                labels = [labels]
            sfreq = float(np.asarray(_deref(f, eeg['srate'])[()]).squeeze())
            count = int(np.asarray(_deref(f, eeg['nbchan'])[()]).squeeze())
            history = _text(f, eeg['history']) if 'history' in eeg else ''
            if include_coordinates:
                coordinates = {axis:[None]*count for axis in ('X','Y','Z')}
                for axis in coordinates:
                    if axis not in locations:
                        continue
                    field = locations[axis]
                    if h5py.check_dtype(ref=field.dtype) is not None:
                        values = [_coordinate(f[ref][()]) if ref else None
                                  for ref in field[()].flatten(order='F')]
                    else:
                        values = [_coordinate(v) for v in field[()].flatten(order='F')]
                    if len(values) != count:
                        raise ValueError(f'Coordinate count differs from channel count: {axis}')
                    coordinates[axis] = values
    else:
        if path.stat().st_size > 512 * 2**20:
            raise ValueError('MAT v5 metadata file exceeds 512 MiB; supply a smaller run or verified header-only export')
        from scipy.io import loadmat
        eeg = loadmat(path, variable_names=['EEG'], struct_as_record=False, squeeze_me=True)['EEG']
        labels = [str(c.labels) for c in np.atleast_1d(eeg.chanlocs)]
        sfreq, count = float(eeg.srate), int(eeg.nbchan)
        history = str(getattr(eeg, 'history', ''))
        if include_coordinates:
            coordinates = {axis:[_coordinate(getattr(c, axis, []))
                                 for c in np.atleast_1d(eeg.chanlocs)]
                           for axis in ('X','Y','Z')}
    if not np.isfinite(sfreq) or sfreq <= 0 or len(labels) != count or not labels or len(set(labels)) != len(labels):
        raise ValueError(f'Invalid channel/sampling metadata: {path}')
    result = {'source': str(path.resolve()), 'sha256': sha256_file(path), 'sfreq': sfreq,
              'channels': labels, 'history': history}
    if include_coordinates:
        result['electrodes'] = [dict(label=label, **{axis:coordinates[axis][i] for axis in coordinates})
                                for i, label in enumerate(labels)]
        result['coordinate_convention'] = 'Stored EEGLAB XYZ; frame, origin and units not independently verified'
    return result
