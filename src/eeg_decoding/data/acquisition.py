"""Public OSF NR acquisition with source-declared SHA256 verification and resumable downloads."""
import json
from pathlib import Path
import re
import urllib.request
from .metadata import sha256_file

NODES = {'zuco1': 'q3zws', 'zuco2': '2urht'}
RESULT = re.compile(r'^results([A-Za-z0-9]+)_NR\.mat$', re.I)
HEADER = re.compile(r'(?:^|_)([A-Za-z0-9]+)_NR\d+_EEG\.mat$', re.I)


def osf_list(node, folder=''):
    url = f'https://api.osf.io/v2/nodes/{node}/files/osfstorage/{folder.strip("/")}'
    if not url.endswith('/'):
        url += '/'
    rows = []
    while url:
        with urllib.request.urlopen(url, timeout=45) as response:
            page = json.load(response)
        rows.extend(page['data'])
        url = page.get('links', {}).get('next')
    return rows


def _folder(rows, predicate, context):
    matches = [r for r in rows if r['attributes']['kind'] == 'folder' and predicate(r['attributes']['name'].lower())]
    if len(matches) != 1:
        raise ValueError(f'Cannot unambiguously locate {context}; inspect official OSF structure')
    return matches[0]['attributes']['path']


def _entry(row, role, subject):
    a = row['attributes']
    name = a['name']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Unsafe OSF filename')
    checksum = a.get('extra', {}).get('hashes', {}).get('sha256')
    if not checksum or not re.fullmatch(r'[0-9a-fA-F]{64}', checksum):
        raise ValueError(f'OSF has no usable SHA256 for {name}; no unchecked download performed')
    return {'name': name, 'subject': subject, 'role': role, 'size': a['size'],
            'sha256': checksum.lower(), 'url': row['links']['download']}


def plan_download(dataset, subjects=None):
    node = NODES[dataset]
    root = osf_list(node)
    task = _folder(root, lambda s: s.replace(' ', '').endswith('-nr'), 'normal-reading task')
    task_rows = osf_list(node, task)
    exports = _folder(task_rows, lambda s: s == 'matlab files', 'MATLAB exports')
    preprocessed = _folder(task_rows, lambda s: s == 'preprocessed', 'preprocessed EEG')
    folders = {r['attributes']['name'].upper(): r['attributes']['path'] for r in osf_list(node, preprocessed)
               if r['attributes']['kind'] == 'folder'}
    entries = []
    found = set()
    requested = {s.upper() for s in subjects} if subjects else None
    for row in osf_list(node, exports):
        match = RESULT.match(row['attributes']['name'])
        if not match:
            continue
        subject = match.group(1).upper()
        if requested and subject not in requested:
            continue
        found.add(subject)
        entries.append(_entry(row, 'sentences', subject))
        if subject not in folders:
            raise ValueError(f'No preprocessed metadata folder for {subject}')
        candidates = [r for r in osf_list(node, folders[subject])
                      if HEADER.search(r['attributes']['name'])
                      and HEADER.search(r['attributes']['name'])[1].upper() == subject
                      and r['attributes'].get('size')]
        if not candidates:
            raise ValueError(f'No EEG metadata recording for {subject}')
        smallest = min(candidates, key=lambda r: r['attributes']['size'])
        entries.append(_entry(smallest, 'metadata', subject))
    if requested and requested - found:
        raise ValueError(f'Unknown requested subjects: {sorted(requested - found)}')
    if not entries:
        raise ValueError('No normal-reading exports found')
    return {'dataset': dataset, 'task': 'nr', 'subjects': sorted(found), 'files': entries,
            'total_bytes': sum(e['size'] for e in entries),
            'metadata_scope': 'one smallest preprocessed run per subject; full cross-run montage audit remains external'}


def download_file(entry, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / entry['name']
    if target.exists():
        if target.stat().st_size == entry['size'] and sha256_file(target) == entry['sha256']:
            return 'reused'
        raise FileExistsError(f'Existing file does not match OSF checksum; preserved without overwrite: {target}')
    partial = target.with_suffix(target.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry['size']:
        raise ValueError(f'Oversized partial download; inspect before retry: {partial}')
    if offset != entry['size']:
        request = urllib.request.Request(entry['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            status = response.status
            if status == 206:
                content_range = response.headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                if not match or int(match[1]) != offset or int(match[3]) != entry['size']:
                    raise ValueError('Unexpected partial response; refusing to append mismatched content')
                mode = 'ab' if offset else 'wb'
            elif status == 200:
                # A server may ignore Range; restart the incomplete file safely.
                mode = 'wb'
            else:
                raise ValueError(f'Unexpected download response {status}')
            with partial.open(mode) as output:
                last_progress = output.tell()
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
                    if output.tell() - last_progress >= 64 * 2**20:
                        last_progress = output.tell()
                        print(f"  {100*last_progress/entry['size']:.0f}% ({last_progress/2**20:.0f} MiB)", flush=True)
    if partial.stat().st_size != entry['size'] or sha256_file(partial) != entry['sha256']:
        raise ValueError(f'Download size/checksum mismatch; file remains quarantined as {partial.name}')
    partial.replace(target)
    return 'downloaded'
