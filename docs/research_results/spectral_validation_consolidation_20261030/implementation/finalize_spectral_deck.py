"""Explicit subtitle style/stacking for the new validation-only slides."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as E
from consolidate_presentation import P,A,NS,xml,shape

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/spectral_consolidation_20261030/NeuroText_spectral_draft.pptx'
OUT=ROOT/'results/spectral_consolidation_20261030/NeuroText_spectral_v2.pptx'
if OUT.exists():raise FileExistsError(OUT)
with ZipFile(SOURCE) as z:parts={n:z.read(n) for n in z.namelist()}
for i in range(21,27):
    root=E.fromstring(parts[f'ppt/slides/slide{i}.xml']);obj=shape(root,4);body=obj.find('p:txBody',NS)
    for child in list(body):body.remove(child)
    pr=E.SubElement(body,f'{{{A}}}bodyPr',wrap='square',lIns='18288',rIns='18288',tIns='18288',bIns='18288',anchor='ctr')
    E.SubElement(pr,f'{{{A}}}noAutofit');E.SubElement(body,f'{{{A}}}lstStyle')
    paragraph=E.SubElement(body,f'{{{A}}}p');props=E.SubElement(paragraph,f'{{{A}}}pPr')
    aft=E.SubElement(props,f'{{{A}}}spcAft');E.SubElement(aft,f'{{{A}}}spcPts',val='0')
    run=E.SubElement(paragraph,f'{{{A}}}r');style=E.SubElement(run,f'{{{A}}}rPr',sz='1600',b='0')
    fill=E.SubElement(style,f'{{{A}}}solidFill');E.SubElement(fill,f'{{{A}}}srgbClr',val='526474')
    E.SubElement(style,f'{{{A}}}latin',typeface='Aptos')
    E.SubElement(run,f'{{{A}}}t').text='VALIDATION ONLY  /  ZuCo1 seed 42  /  512 queries  /  298 candidates'
    tree=root.find('p:cSld/p:spTree',NS);tree.remove(obj);tree.append(obj)
    parts[f'ppt/slides/slide{i}.xml']=xml(root)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
print(OUT)
