"""Extend the existing deck's editable OOXML using stored metrics only."""
import copy
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as E
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/presentation/archive/pre_wider_grid/NeuroText_draft.pptx'
OUT=ROOT/'docs/presentation/NeuroText_consolidated.pptx'
DATA=ROOT/'docs/research_results/wider_grid_transfer'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
NS={'p':P,'a':A,'r':R}
for prefix,uri in NS.items(): E.register_namespace(prefix,uri)


def xml(node):
    if node.tag.startswith('{'+PKG+'}'): E.register_namespace('',PKG)
    if node.tag.startswith('{'+CT+'}'): E.register_namespace('',CT)
    return E.tostring(node,encoding='utf-8',xml_declaration=True)
def emu(inches): return str(round(inches*914400))


def shape(root, identifier):
    return next(s for s in root.findall('.//p:sp',NS)
                if s.find('p:nvSpPr/p:cNvPr',NS).get('id')==str(identifier))


def replace_text(s, content, size=None):
    body=s.find('p:txBody',NS)
    template=copy.deepcopy(body.find('a:p',NS))
    for old in body.findall('a:p',NS): body.remove(old)
    for line in content.split('\n'):
        paragraph=copy.deepcopy(template)
        for node in list(paragraph):
            if node.tag!=f'{{{A}}}pPr': paragraph.remove(node)
        r=E.SubElement(paragraph,f'{{{A}}}r')
        style=template.find('a:r/a:rPr',NS)
        if style is not None: r.append(copy.deepcopy(style))
        elif size is not None: E.SubElement(r,f'{{{A}}}rPr')
        if size is not None:
            props=r.find('a:rPr',NS)
            if props is None: props=E.SubElement(r,f'{{{A}}}rPr')
            props.set('sz',str(round(size*100)))
            ppr=paragraph.find('a:pPr',NS)
            if ppr is not None and ppr.get('sz'): ppr.set('sz',str(round(size*100)))
        E.SubElement(r,f'{{{A}}}t').text=line
        body.append(paragraph)


def notes(raw, text):
    root=E.fromstring(raw)
    s=next(s for s in root.findall('.//p:sp',NS)
           if s.find('p:nvSpPr/p:nvPr/p:ph',NS) is not None
           and s.find('p:nvSpPr/p:nvPr/p:ph',NS).get('type')=='body')
    replace_text(s,text)
    return xml(root)


def add_table(root, rows, widths, size=18):
    tree=root.find('p:cSld/p:spTree',NS)
    tree.remove(shape(root,7))  # source body box, replaced with an editable table
    frame=E.SubElement(tree,f'{{{P}}}graphicFrame')
    nv=E.SubElement(frame,f'{{{P}}}nvGraphicFramePr')
    E.SubElement(nv,f'{{{P}}}cNvPr',id='20',name='Audited results table')
    E.SubElement(nv,f'{{{P}}}cNvGraphicFramePr'); E.SubElement(nv,f'{{{P}}}nvPr')
    xf=E.SubElement(frame,f'{{{P}}}xfrm')
    E.SubElement(xf,f'{{{A}}}off',x=emu(.75),y=emu(1.85))
    E.SubElement(xf,f'{{{A}}}ext',cx=emu(sum(widths)),cy=emu(4.1))
    graphic=E.SubElement(frame,f'{{{A}}}graphic')
    data=E.SubElement(graphic,f'{{{A}}}graphicData',uri='http://schemas.openxmlformats.org/drawingml/2006/table')
    table=E.SubElement(data,f'{{{A}}}tbl')
    E.SubElement(table,f'{{{A}}}tblPr',firstRow='1',bandRow='1')
    grid=E.SubElement(table,f'{{{A}}}tblGrid')
    for width in widths: E.SubElement(grid,f'{{{A}}}gridCol',w=emu(width))
    for i,row in enumerate(rows):
        tr=E.SubElement(table,f'{{{A}}}tr',h=emu(4.1/len(rows)))
        for value in row:
            tc=E.SubElement(tr,f'{{{A}}}tc')
            body=E.SubElement(tc,f'{{{A}}}txBody')
            E.SubElement(body,f'{{{A}}}bodyPr'); E.SubElement(body,f'{{{A}}}lstStyle')
            paragraph=E.SubElement(body,f'{{{A}}}p')
            E.SubElement(paragraph,f'{{{A}}}pPr',algn='l')
            run=E.SubElement(paragraph,f'{{{A}}}r')
            props=E.SubElement(run,f'{{{A}}}rPr',sz=str(size*100),b='1' if i==0 else '0')
            fill=E.SubElement(props,f'{{{A}}}solidFill')
            E.SubElement(fill,f'{{{A}}}srgbClr',val='FFFFFF' if i==0 else '142D44')
            E.SubElement(props,f'{{{A}}}latin',typeface='Aptos')
            E.SubElement(run,f'{{{A}}}t').text=str(value)
            prop=E.SubElement(tc,f'{{{A}}}tcPr',marL=emu(.10),marR=emu(.06),marT=emu(.04),marB=emu(.03),anchor='ctr')
            fill=E.SubElement(prop,f'{{{A}}}solidFill')
            E.SubElement(fill,f'{{{A}}}srgbClr',val='087F8C' if i==0 else ('F1F5F8' if i%2 else 'FFFFFF'))


def build(output=OUT):
    output=Path(output)
    if not output.is_absolute(): output=ROOT/output
    if output.exists(): raise FileExistsError('Use a fresh output deck; preserve the existing revision')
    summary=json.loads((DATA/'summary.json').read_text())
    with ZipFile(BASE) as source: parts={name:source.read(name) for name in source.namelist()}
    # Keep the source theme, masters, layouts, fonts, and historical chart slides.
    first=E.fromstring(parts['ppt/slides/slide1.xml'])
    for s in first.findall('.//p:sp',NS):
        text=''.join(s.itertext())
        if 'Executed within-dataset results;' in text:
            replace_text(s,'18 wider-grid exploratory transfer evaluations; participant independence unresolved.',16)
    parts['ppt/slides/slide1.xml']=xml(first)
    parts['ppt/notesSlides/notesSlide1.xml']=notes(parts['ppt/notesSlides/notesSlide1.xml'],
        'Source: docs/research_results/REPORT.md. Original 79-report export retained. Added 18 '
        'wider-grid exploratory transfer evaluations, with six original-grid seed42 comparisons '
        'kept separate. Report counts are not independent methods or cohorts. Strict cross-release '
        'participant independence remains unverified.')
    source_note='Source: docs/research_results/wider_grid_transfer/summary.json and configurations_and_pools.json. '
    specs={
      13:dict(title='Exploratory transfer uses source-only selection',subtitle='18 wider-grid evaluations · both directions · seeds 42, 43, 44',
          body='Train signal normalization and ridge on source training observations only.\n'
               'Select alpha by source-validation MRR across seven fixed values.\n'
               'Compare real EEG with separately trained Gaussian and shuffled controls.\n'
               'Participant independence across releases remains unresolved.',
          callout='50 shared texts remain in the target pools; strict transfer is still unexecuted.',
          notes=source_note+'Grid .1,1,10,100,1000,10000,100000. Full recordings, 105 channels, 200 Hz, NR. '
                'Source-only validation chooses alpha, first maximum wins ties. No target fitting. '
                'All reports use transfer_exploratory and identity_audited:false. Different release '
                'labels do not verify different people. Inclusive target pools contain 50 shared texts.'),
      14:dict(title='Wider-grid results: ZuCo1 to ZuCo2',subtitle='5,884 queries · 344 candidates · the same target cohort at every seed',
          callout='Real EEG MRR is below Gaussian MRR on all three seeds.'),
      15:dict(title='Wider-grid results: ZuCo2 to ZuCo1',subtitle='3,349 queries · 300 candidates · the same target cohort at every seed',
          callout='Real EEG MRR exceeds Gaussian on seeds 42 and 43; the sign reverses on seed 44.'),
      16:dict(title='Wider-grid means remain close to controls',subtitle='Equal-weight seed means · sample SD · individual seed points',
          figure='wider_grid_mean_sd.png',callout='Repeated splits reuse target participants and queries; SD is descriptive.',
          notes=source_note+'Mean MRR ± sample SD: forward real .018529±.000603, Gaussian .019346±.001414, '
                'shuffled .018247±.000494; reverse .023592±.001743, .023142±.000736, .021216±.000291. '
                'SD uses ddof=1 across seeds. It is not a CI or standard error. Dashed lines are '
                'uniform ranking expectations, not significance thresholds. Gaussian preserves amplitude and duration.'),
      17:dict(title='Real-minus-Gaussian differences change by seed',subtitle='Pair within direction and seed before calculating mean and sample SD',
          figure='wider_grid_real_minus_gaussian.png',callout='Forward ΔMRR −0.000817 ± 0.000831; reverse +0.000450 ± 0.002237.',
          notes=source_note+'Forward paired MRR differences: -.000774665, -.001667989, -.000008301. '
                'Reverse: +.001579119, +.001898438, -.002126646. Positive reverse differences in 2/3 '
                'seeds; forward 0/3. Same target cohorts and overlapping source splits. No independent-cohort '
                'or significance claim. Mean ΔRecall@1 pp: forward -.067981±.148161; reverse +.049766±.198815.'),
      18:dict(title='Original and wider grids are separate experiments',subtitle='Seed 42 only · original grid ends at 100 · wider grid ends at 100,000',
          callout='Only the wider grid has seeds 42–44; do not pool the grids as extra repeats.'),
      19:dict(title='Archive and recorded software validation',subtitle='139 passed · 2 optional skips · 0 deselected',
          body='Full configs, candidate keys, split digests and evaluation JSON are archived.\n'
               'Two official-weight tests skip because EEG_LABRAM_CACHE is unset.\n'
               'Integration skip: no network download during unit tests.\n'
               'Extraction skip: optional official weights on synthetic fixtures.\n'
               'The previously excluded 513 MiB MATLAB test was included and passed.',
          callout='Recorded validation is preserved; consolidation launches no experiments.',
          notes=source_note+'Recorded command: .venv/Scripts/python.exe -m pytest -q -rs. '
                '139 passed, 2 skipped, 0 deselected, two existing short-signal matplotlib warnings. '
                'Exact skips: tests/test_storage_and_pretrained.py::test_official_pretrained_weights_on_synthetic_input '
                '(optional official-weight integration; no network download during unit tests) and '
                '::test_pretrained_cache_extraction (optional official-weight extraction on synthetic fixtures). '
                'EEG_LABRAM_CACHE unset. tests/test_mat5_streaming.py::test_file_above_old_limit_without_full_materialization '
                'was included and passed. Archive: archive_manifest.json, runs.csv, condition_summary.csv, '
                'paired_differences.csv, difference_summary.csv, configs/, implementation/, evaluations/.'),
      20:dict(title='These configurations do not show reliable retrieval',subtitle='Matched controls and repeated splits limit the interpretation',
          body='These configurations do not demonstrate reliable retrieval beyond controls.\n'
               'Forward real MRR trails Gaussian on all three seeds.\n'
               'Reverse differences change sign, with only a small positive mean.\n'
               'Participant independence remains unresolved; strict transfer is unexecuted.',
          callout='The findings do not establish that EEG contains no semantic information.',
          notes=source_note+'Conclusion concerns these tested configurations, not all EEG decoding. '
                'Same target participants and queries at every seed, overlapping source splits, shared '
                'sentences, source timing/reference uncertainties and exploratory test inspection limit '
                'inference. Original 79-row export and original-grid transfer remain separate. No new '
                'experiments were run during consolidation. Independent identity evidence and protocol-matched '
                'paper reproduction remain unresolved.')}
    original=list(csv.DictReader((DATA/'original_grid_seed42.csv').open()))
    content_types=E.fromstring(parts['[Content_Types].xml'])
    presentation=E.fromstring(parts['ppt/presentation.xml'])
    relations=E.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    slide_list=presentation.find('p:sldIdLst',NS)
    max_id=max(int(s.get('id')) for s in slide_list)
    for number,spec in specs.items():
        base=6 if 'figure' in spec else 13
        root=E.fromstring(parts[f'ppt/slides/slide{base}.xml']) if base==6 else E.fromstring(ZipFile(BASE).read('ppt/slides/slide13.xml'))
        replace_text(shape(root,3),spec['title'],30)
        replace_text(shape(root,4),spec['subtitle'],16)
        replace_text(shape(root,6),f'{number:02}',10)
        replace_text(shape(root,9),spec['callout'],16)
        if 'body' in spec: replace_text(shape(root,7),spec['body'],22 if number in [19,20] else 24)
        if number in [14,15]:
            direction='zuco1_to_zuco2' if number==14 else 'zuco2_to_zuco1'
            rows=[['Seed','Condition','Alpha','Source val MRR','Recall@1','Target MRR']]
            for seed in [42,43,44]:
                for control in ['real','gaussian','shuffled']:
                    r=next(r for r in summary['runs'] if (r['direction'],r['seed'],r['control'])==(direction,seed,control))
                    rows.append([seed,control.capitalize(),f"{r['selected_alpha']:g}",f"{r['validation_mrr']:.6f}",f"{100*r['recall_at_1']:.3f}%",f"{r['mrr']:.6f}"])
            add_table(root,rows,[.95,2.2,1.7,2.35,2.0,2.6])
            spec['notes']=source_note+'All nine seed-level results and source-validation winners are shown. '
            spec['notes']+='\n'+json.dumps([r for r in summary['runs'] if r['direction']==direction])
        if number==18:
            rows=[['Direction','Condition','Original α / MRR','Wider α / MRR']]
            for d in ['zuco1_to_zuco2','zuco2_to_zuco1']:
                for c in ['real','gaussian','shuffled']:
                    old=next(r for r in original if r['direction']==d and r['control']==c)
                    new=next(r for r in summary['runs'] if (r['direction'],r['seed'],r['control'])==(d,42,c))
                    rows.append(['1 to 2' if d.startswith('zuco1') else '2 to 1',c.capitalize(),
                                 f"{float(old['selected_alpha']):g} / {float(old['mrr']):.6f}",
                                 f"{new['selected_alpha']:g} / {new['mrr']:.6f}"])
            add_table(root,rows,[2,2.2,3.8,3.8],18)
            spec['notes']='Source: docs/research_results/wider_grid_transfer/original_grid_seed42.csv and summary.json. '
            spec['notes']+='Original grid [.1,1,10,100], wider grid [.1,1,10,100,1000,10000,100000]. '
            spec['notes']+='Original grid has only seed42 transfer; no three-seed grid-effect claim. Reverse real R1 .328% to .627% at seed42. '
            spec['notes']+='Keep separate from original 79-report within-dataset export. No target score selected alpha.'
        rels=E.fromstring(ZipFile(BASE).read(f'ppt/slides/_rels/slide{base}.xml.rels'))
        for rel in rels:
            if rel.get('Type').endswith('/notesSlide'): rel.set('Target',f'../notesSlides/notesSlide{number}.xml')
            if rel.get('Type').endswith('/image') and 'figure' in spec:
                rel.set('Target','../media/'+spec['figure']); parts['ppt/media/'+spec['figure']]=(DATA/spec['figure']).read_bytes()
        if 'figure' in spec:
            image=root.find('.//p:pic/p:spPr/a:xfrm',NS)
            with Image.open(DATA/spec['figure']) as im: ratio=im.width/im.height
            w=min(12,4.32*ratio); h=w/ratio
            image.find('a:off',NS).set('x',emu(.65+(12-w)/2)); image.find('a:off',NS).set('y',emu(1.75+(4.32-h)/2))
            image.find('a:ext',NS).set('cx',emu(w)); image.find('a:ext',NS).set('cy',emu(h))
        parts[f'ppt/slides/slide{number}.xml']=xml(root)
        parts[f'ppt/slides/_rels/slide{number}.xml.rels']=xml(rels)
        parts[f'ppt/notesSlides/notesSlide{number}.xml']=notes(ZipFile(BASE).read('ppt/notesSlides/notesSlide13.xml'),spec['notes'])
        nr=E.fromstring(ZipFile(BASE).read('ppt/notesSlides/_rels/notesSlide13.xml.rels'))
        for rel in nr:
            if rel.get('Type').endswith('/slide'): rel.set('Target',f'../slides/slide{number}.xml')
        parts[f'ppt/notesSlides/_rels/notesSlide{number}.xml.rels']=xml(nr)
        if number>14:
            rid=f'rId{number+7}'
            E.SubElement(relations,f'{{{PKG}}}Relationship',Id=rid,Type=f'{R}/slide',Target=f'slides/slide{number}.xml')
            E.SubElement(slide_list,f'{{{P}}}sldId',id=str(max_id+number-14),attrib={f'{{{R}}}id':rid})
            for stem,mime in [('slides/slide','slide'),('notesSlides/notesSlide','notesSlide')]:
                E.SubElement(content_types,f'{{{CT}}}Override',PartName=f'/ppt/{stem}{number}.xml',
                             ContentType=f'application/vnd.openxmlformats-officedocument.presentationml.{mime}+xml')
    parts['ppt/presentation.xml']=xml(presentation)
    parts['ppt/_rels/presentation.xml.rels']=xml(relations)
    parts['[Content_Types].xml']=xml(content_types)
    app=E.fromstring(parts['docProps/app.xml'])
    for prop in app:
        if prop.tag.rsplit('}',1)[-1] in ['Slides','Notes']: prop.text='20'
    parts['docProps/app.xml']=xml(app)
    with ZipFile(output,'w',ZIP_DEFLATED) as target:
        for name,value in parts.items(): target.writestr(name,value)
    print(f'Saved 20-slide consolidated deck: {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(OUT))
    build(parser.parse_args().output)
