"""Make validation scope explicit in every new title, independent of subtitles."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import xml.etree.ElementTree as E
from consolidate_presentation import NS,xml,shape,replace_text

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'results/spectral_consolidation_20261030'
target=BUILD/'NeuroText_spectral_final.pptx'
if target.exists():raise FileExistsError(target)
with ZipFile(BUILD/'NeuroText_spectral_v2.pptx') as z:parts={n:z.read(n) for n in z.namelist()}
titles=['Validation-only TRT spectral pilot','Validation-only pilot favors structural controls',
        'Validation-only incremental comparison','Validation-only incremental results',
        'Validation-only reader effects change sign','Validation-only results show no consistent EEG gain']
for i,title in enumerate(titles,21):
    root=E.fromstring(parts[f'ppt/slides/slide{i}.xml'])
    replace_text(shape(root,3),title,30)
    replace_text(shape(root,4),'',16)
    if i==21:
        replace_text(shape(root,7),'TRT pools eight 105-channel word vectors into sentence features.\n'
            'The cohort has 2,245 training observations and 512 validation queries.\n'
            'All validation scores rank the same 298 sentence candidates.\n'
            'Counts and missingness require explicit structural controls.',22)
    parts[f'ppt/slides/slide{i}.xml']=xml(root)
with ZipFile(target,'w',ZIP_DEFLATED) as z:
    for n,data in parts.items():z.writestr(n,data)
print(target)
