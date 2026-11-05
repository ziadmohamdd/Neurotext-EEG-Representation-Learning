"""Document-only checks. Never import a model, data adapter or evaluator."""
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shutil
import xml.etree.ElementTree as E
from zipfile import ZipFile
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'results/spectral_consolidation_20261030'
EVIDENCE=ROOT/'docs/research_results/spectral_validation_consolidation_20261030'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    raw=Path(path).read_bytes()
    return json.loads(raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'))


def table_rows(z,number):
    root=E.fromstring(z.read(f'ppt/slides/slide{number}.xml'))
    return [[''.join(t.text or '' for t in c.findall('.//a:t',NS)) for c in row.findall('a:tc',NS)] for row in root.findall('.//a:tbl/a:tr',NS)]


def main():
    previous=read_json(BUILD/'preservation_before.json')
    assert all(sha(ROOT/p)==digest for p,digest in previous.items())
    archive=ROOT/'docs/presentation/archive/pre_spectral_20261030'
    archived=read_json(archive/'manifest.json')+read_json(archive/'additional_preservation_manifest.json')
    for row in archived:assert sha(ROOT/row['archive'])==row['sha256']
    evidence=read_json(EVIDENCE/'stored_evidence.json');pilot=evidence['pilot_selected_models'];inc=evidence['incremental_selected_models'];diff=evidence['incremental_differences']
    assert all(r['evaluation_split']=='val' and r['metrics']['n_queries']==512 and len(r['candidate_keys'])==298 for r in pilot+inc)
    assert all(r['candidate_keys']==pilot[0]['candidate_keys'] and r['observation_identities']==pilot[0]['observation_identities'] for r in pilot+inc)
    deck=ROOT/'docs/presentation/NeuroText_consolidated.pptx';pdf=ROOT/'docs/presentation/NeuroText_consolidated.pdf'
    assert sha(deck)==sha(BUILD/'NeuroText_spectral_final.pptx')
    assert sha(pdf)==sha(BUILD/'render_final/preview.pdf')
    with ZipFile(deck) as z,ZipFile(archive/deck.name) as old:
        assert z.testzip() is None
        assert len(E.fromstring(z.read('ppt/presentation.xml')).find('p:sldIdLst',NS))==26
        assert E.fromstring(z.read('ppt/presentation.xml')).find('p:sldSz',NS).attrib==E.fromstring(old.read('ppt/presentation.xml')).find('p:sldSz',NS).attrib
        for name in z.namelist():
            if name.endswith(('.xml','.rels')):E.fromstring(z.read(name))
            if name.endswith('.rels'):
                base='' if name=='_rels/.rels' else name.rsplit('/_rels/',1)[0]
                for rel in E.fromstring(z.read(name)):
                    if rel.get('TargetMode')!='External':assert posixpath.normpath(posixpath.join(base,rel.get('Target'))).lstrip('/') in z.namelist()
        for i in range(2,21):
            left=E.fromstring(old.read(f'ppt/slides/slide{i}.xml'));right=E.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
            for t in left.findall('.//a:t',NS):
                if t.text:t.text=t.text.replace('29 OCT 2026','30 OCT 2026')
            assert E.tostring(left)==E.tostring(right)
            a=Image.open(BUILD/f'render_source/Slide{i}.PNG').convert('RGB').crop((0,0,1280,680))
            b=Image.open(BUILD/f'render_final/Slide{i}.PNG').convert('RGB').crop((0,0,1280,680))
            assert ImageChops.difference(a,b).getbbox() is None
        for name in old.namelist():
            if name.startswith(('ppt/media/','ppt/theme/','ppt/slideMasters/','ppt/slideLayouts/')):assert old.read(name)==z.read(name)
        for i in range(21,27):
            root=E.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
            title=next(s for s in root.findall('.//p:sp',NS) if s.find('p:nvSpPr/p:cNvPr',NS).get('id')=='3')
            assert 'Validation-only' in ''.join(title.itertext())
            notes=E.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml'))
            assert 'Source' in ''.join(notes.itertext()) or 'Sources' in ''.join(notes.itertext())
        labels={'real_trt':'Real TRT','matched_gaussian':'Matched Gaussian','class_shuffled':'Class shuffled','count_duration_only':'Count / duration','zero_missingness_only':'Zero / missingness','existing_sentence_ridge':'Sentence ridge'}
        rows=table_rows(z,22);assert len(rows)==7
        for row,r in zip(rows[1:],pilot):
            m=r['metrics'];assert row==[labels[r['condition']],f"{100*m['recall@1']:.3f}%",f"{100*m['recall@5']:.3f}%",f"{100*m['recall@10']:.3f}%",f"{m['mrr']:.6f}"]
        rows=table_rows(z,24);assert len(rows)==4
        for row,r,label in zip(rows[1:],inc,['Structure','Structure + real','Structure + Gaussian']):
            m=r['metrics'];assert row==[label,f"{100*m['recall@1']:.3f}%",f"{100*m['recall@5']:.3f}%",f"{100*m['recall@10']:.3f}%",f"{m['mrr']:.6f}",str(r['selection']['eeg_weight'])]
        assert len(table_rows(z,25))==3
        for row,p in zip(table_rows(z,25)[1:],inc[1]['per_subject']):
            assert row==[p.split(':')[1],str(inc[1]['per_subject'][p]['n_queries']),f"{diff[0]['per_subject'][p]['mrr']:+.6f}",f"{diff[1]['per_subject'][p]['mrr']:+.6f}"]
    raw=pdf.read_bytes();assert raw.startswith(b'%PDF-1.7') and len(re.findall(rb'/Type\s*/Page\b',raw))==26
    report=ROOT/'docs/research_results/REPORT.md';text=report.read_text(encoding='utf-8')
    for r in pilot+inc:
        m=r['metrics']
        assert f"{m['mrr']:.6f}" in text
        for k in ('recall@1','recall@5','recall@10'):assert f"{100*m[k]:.4f}%" in text
    assert '196 settings' in text and 'Repeated validation use' in text and 'not an independent replication' in text
    assert 'Further tuning and test evaluation are stopped' in text
    for p in [report,ROOT/'docs/presentation/README.md',ROOT/'docs/presentation_outline.md',ROOT/'docs/continue_work.md',EVIDENCE/'README.md']:
        s=p.read_text(encoding='utf-8');assert '\ufffd' not in s
        for target in re.findall(r'\]\(([^)]+)\)',s):
            if '://' not in target:assert (p.parent/target).resolve().exists(),target
    geometry=read_json(BUILD/'final_geometry.json');package=read_json(BUILD/'final_package_integrity.json')
    assert geometry['finding_count']==0 and package['finding_count']==0
    # Preserve this consolidation implementation and exact input identities.
    copies=[]
    for source in [ROOT/'scripts/consolidate_spectral_validation.py',ROOT/'scripts/finalize_spectral_deck.py',ROOT/'scripts/make_spectral_final_titles.py',Path(__file__),
                   ROOT/'results/spectral_trt_pilot_v1_20261030/plan.json',ROOT/'results/trt_incremental_v1_20261030/plan.json']:
        folder=EVIDENCE/('implementation' if source.suffix=='.py' else ('spectral_pilot' if 'spectral_trt_pilot' in str(source) else 'incremental'))
        folder.mkdir(exist_ok=True);target=folder/source.name
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(source,target);assert sha(source)==sha(target)
        copies.append({'source':str(source.relative_to(ROOT)),'copy':str(target.relative_to(ROOT)),'sha256':sha(source)})
    (EVIDENCE/'archive_manifest.json').write_text(json.dumps(copies,indent=2)+'\n',encoding='utf-8')
    result={'scope':'Consolidation artifact checks only. No fitting, tuning, feature extraction, model or test evaluation.',
        'status':'passed','slides':26,'pdf_pages':26,'new_validation_only_slides':[21,22,23,24,25,26],
        'score_rows_checked':9,'common_validation_queries':512,'common_candidates':298,'native_tables_checked':[22,24,25],
        'historical_slides_2_to_20':'Source XML identical except date; rendered pixels identical outside date footer',
        'template_assets':'Original media, layouts, masters and theme byte-identical','native_package_findings':0,'geometry_findings':0,
        'powerpoint_text_overflow_findings':0,'visual_review':'Cover and six new slides inspected at full size; historical slides compared against fresh source renders',
        'pdf_verification':'Native PowerPoint export; PDF signature and 26 page objects verified. No optional PDF text-reader dependency installed.',
        'prior_protected_files_unchanged':len(previous),'previous_report_deck_pdf_and_support_docs_archived':len(archived),
        'model_tests_run_during_consolidation':False,'deliverables':{str(p.relative_to(ROOT)):sha(p) for p in [report,deck,pdf]}}
    (EVIDENCE/'consolidation_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
