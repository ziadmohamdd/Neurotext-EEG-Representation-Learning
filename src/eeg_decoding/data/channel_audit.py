"""Export actual electrode coordinates for a positional mapping audit; no mapping is guessed."""
import argparse
from pathlib import Path

from .metadata import read_eeg_metadata
from ..utils.logging import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = {'status':'coordinates_exported_not_mapping_verified', 'recordings':[], 'errors':[]}
    paths = sorted({p.resolve() for root in args.roots
                    for p in Path(root).rglob('*_EEG.mat')})
    if not paths:
        raise SystemExit('No *_EEG.mat recording headers found in the supplied roots')
    for path in paths:
        try:
            metadata = read_eeg_metadata(path, include_coordinates=True)
            complete = sum(all(e[axis] is not None for axis in ('X','Y','Z'))
                           for e in metadata['electrodes'])
            metadata['n_complete_xyz'] = complete
            report['recordings'].append(metadata)
            print(f'{path.name}: {complete}/{len(metadata["channels"])} channels have XYZ', flush=True)
        except Exception as exc:
            report['errors'].append({'source':str(path), 'error':str(exc)})
            print(f'{path.name}: FAILED: {exc}', flush=True)
    if report['errors']:
        report['status'] = 'partial_coordinate_export'
    write_json(args.output, report)
    print(f'Coordinate audit saved: {args.output}', flush=True)
    if report['errors']:
        raise SystemExit('Some headers failed; see the saved errors before positional mapping')


if __name__ == '__main__':
    main()
