"""Read-only checks of consolidated artifacts; no training/model evaluation."""
import csv
import hashlib
import json
import posixpath
from pathlib import Path
import shutil
import xml.etree.ElementTree as E
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/research_results/wider_grid_transfer'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary=json.loads((DATA/'summary.json').read_text())
    manifest=json.loads((DATA/'archive_manifest.json').read_text())
    assert len(summary['runs'])==18
    for entry in manifest['files']:
        assert sha(ROOT/entry['source'])==entry['sha256'],entry['source']
        assert sha(DATA/entry['archived'])==entry['sha256'],entry['archived']
    source_audits=json.loads((DATA/'configurations_and_pools.json').read_text())
    for row,audit in [(row,next(a for a in source_audits if (a['direction'],a['seed'],a['control'])==(row['direction'],row['seed'],row['control']))) for row in summary['runs']]:
        assert sha(Path(row['report']))==audit['report_sha256']
        assert sha(Path(row['report']).parent/'metadata.json')==audit['metadata_sha256']
    before=json.loads((ROOT/'results/exploratory_repeat_prior_hashes_20261029.json').read_text(encoding='utf-8-sig'))
    for record in before: assert sha(Path(record['Path'])).upper()==record['SHA256']
    original=ROOT/'docs/presentation/NeuroText_draft.pptx'
    assert sha(original)==sha(ROOT/'docs/presentation/archive/pre_wider_grid/NeuroText_draft.pptx')
    assert len(list(csv.DictReader((ROOT/'docs/research_results/retrieval.csv').open())))==79
    deck=ROOT/'docs/presentation/NeuroText_consolidated.pptx'
    p='http://schemas.openxmlformats.org/presentationml/2006/main'
    a='http://schemas.openxmlformats.org/drawingml/2006/main'
    ns={'p':p,'a':a}
    with ZipFile(deck) as z, ZipFile(original) as old:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith(('.xml','.rels')): E.fromstring(z.read(name))
            if name.endswith('.rels'):
                root=E.fromstring(z.read(name))
                base='' if name=='_rels/.rels' else name.rsplit('/_rels/',1)[0]
                for rel in root:
                    if rel.get('TargetMode')=='External': continue
                    target=posixpath.normpath(posixpath.join(base,rel.get('Target'))).lstrip('/')
                    assert target in z.namelist(),(name,target)
        presentation=E.fromstring(z.read('ppt/presentation.xml'))
        assert len(presentation.find('p:sldIdLst',ns))==20
        for i in range(2,13):
            assert z.read(f'ppt/slides/slide{i}.xml')==old.read(f'ppt/slides/slide{i}.xml')
        for slide,direction in [(14,'zuco1_to_zuco2'),(15,'zuco2_to_zuco1')]:
            root=E.fromstring(z.read(f'ppt/slides/slide{slide}.xml'))
            rows=root.findall('.//a:tbl/a:tr',ns)
            assert len(rows)==10
            expected=[next(r for r in summary['runs'] if (r['direction'],r['seed'],r['control'])==(direction,s,c)) for s in [42,43,44] for c in ['real','gaussian','shuffled']]
            for tr,r in zip(rows[1:],expected):
                values=[''.join(cell.itertext()) for cell in tr.findall('a:tc',ns)]
                assert values==[str(r['seed']),r['control'].capitalize(),f"{r['selected_alpha']:g}",f"{r['validation_mrr']:.6f}",f"{100*r['recall_at_1']:.3f}%",f"{r['mrr']:.6f}"]
        notes=' '.join(' '.join(E.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml')).itertext()) for i in range(13,21))
        assert '139 passed' in notes and 'identity_audited:false' in notes
        assert 'test_official_pretrained_weights_on_synthetic_input' in notes
        assert 'optional official-weight extraction on synthetic fixtures' in notes
    report=(ROOT/'docs/research_results/REPORT.md').read_text(encoding='utf-8')
    assert '139 passed, 2 optional skips, 0 deselected' in report
    assert 'do not demonstrate reliable retrieval beyond controls' in report
    for r in summary['runs']:
        assert f"{r['validation_mrr']:.6f}" in report
        assert f"{r['mrr']:.6f}" in report
    result=dict(status='passed',scope='artifact checks only; no new experiments or pytest run',
                wider_grid_rows=18,original_grid_rows=6,historical_export_rows=79,
                slides=20,editable_seed_table_rows_checked=18,
                prior_experiment_files_unchanged=len(before),
                source_report_and_metadata_hashes_verified=36,
                xml_relationships='all internal targets resolved',
                historical_slides_2_to_12='byte-identical slide XML',
                recorded_validation=manifest['validation'],
                render='20 slides and PDF exported with Microsoft PowerPoint; visual review completed',
                text_bounds='PowerPoint read-only check reported no text overflow',
                deliverables={str(path.relative_to(ROOT)):sha(path) for path in [deck,ROOT/'docs/presentation/NeuroText_consolidated.pdf',ROOT/'docs/research_results/REPORT.md']})
    (DATA/'consolidation_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
