"""Consolidate stored validation evidence into report/deck. No model imports."""
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as E
from zipfile import ZipFile,ZIP_DEFLATED

from consolidate_presentation import P,A,R,PKG,CT,NS,shape,replace_text,notes,xml,add_table

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'results/spectral_consolidation_20261030'
BEFORE=ROOT/'docs/presentation/archive/pre_spectral_20261030'


def rows_table(reports):
    out=['| Condition | Validation Recall@1 | Validation Recall@5 | Validation Recall@10 | Validation MRR |',
         '|---|---:|---:|---:|---:|']
    for r in reports:
        m=r['metrics'];out.append(f"| {r['condition']} | {100*m['recall@1']:.4f}% | {100*m['recall@5']:.4f}% | {100*m['recall@10']:.4f}% | {m['mrr']:.6f} |")
    return out


def main():
    pilot=json.loads((ROOT/'results/spectral_trt_pilot_v1_20261030/validation_summary.json').read_text())
    inc=json.loads((ROOT/'results/trt_incremental_v1_20261030/validation_summary.json').read_text())
    diffs=json.loads((ROOT/'results/trt_incremental_v1_20261030/differences.json').read_text())
    pby={r['condition']:r for r in pilot};iby={r['condition']:r for r in inc}
    assert len(pilot)==6 and len(inc)==3
    assert all(r['evaluation_split']=='val' and r['metrics']['n_queries']==512 and len(r['candidate_keys'])==298 for r in pilot+inc)
    assert all(r['candidate_keys']==pilot[0]['candidate_keys'] and r['observation_identities']==pilot[0]['observation_identities'] for r in pilot+inc)
    assert pby['count_duration_only']['metrics']==iby['structural_only']['metrics']
    evidence=ROOT/'docs/research_results/spectral_validation_consolidation_20261030'
    if evidence.exists():raise FileExistsError(evidence)
    evidence.mkdir()
    combined={'protocol':'validation_only','dataset':'zuco1','seed':42,'training_observations':2245,'validation_observations':512,
        'validation_candidates':298,'pilot_selected_models':pilot,'incremental_selected_models':inc,
        'incremental_differences':diffs,'cohort_interpretation':'Same reused validation cohort; structural baseline repeated across tables is not an independent replication',
        'test_evaluated':False,'new_fits_or_tuning_during_consolidation':False}
    (evidence/'stored_evidence.json').write_text(json.dumps(combined,indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(Path(__file__),evidence/'consolidate_spectral_validation.py')
    section=['## Validation-only TRT spectral and incremental-information experiments','',
        '**All scores in this section are validation-only model-selection results.** They are not test '
        'evaluations and are not appended to the original 79-report retrieval CSV or the wider-grid transfer tables. '
        'The same ZuCo1 seed-42 cohort is reused: 2,245 training observations, 512 validation queries from '
        'ZJS/ZKW and 298 candidates. Sentence classes recur across readers. There are no new exclusions. '
        'The all-missing training observation ZMG/70 is retained with training-fitted imputation. '
        'Original duration-QC exclusions remain unchanged; no test features were extracted for these pilots.','',
        '### Structural retrieval performance is a separate finding','',
        'The **count/duration structural baseline** reaches validation Recall@1/5/10 of '
        '**17.7734% / 40.6250% / 51.3672%**, with **MRR 0.292012**. It uses 14 count/duration summaries '
        'and 48 training-derived word-count basis columns. It has **no measured TRT amplitudes**. '
        'This is performance from annotation-derived reading structure, not evidence that EEG amplitudes '
        'retrieve meaning. Eight candidate sentences have unique word counts and count buckets contain '
        'at most 16 of the 298 candidates. No word text or sentence identity is supplied as a model input, '
        'but this strong structural shortcut remains.','',
        '### Spectral pilot: validation-only selected results','',
        'The new TRT view pools eight 105-channel source word vectors into 840 sentence amplitude means. '
        'The spectral arms also receive three count/missingness summaries and the training-derived word-count '
        'basis (891 features). Count/duration-only has 62 inputs. Zero/missingness-only retains structure '
        'without amplitude information. Gaussian values preserve word counts and exact missingness patterns, '
        'with marginal moments fitted on training word values only. Class shuffling consistently deranges '
        'training sentence labels while validation labels remain correct.','',*rows_table(pilot),'',
        'Each pilot arm selected from **28 settings**: two imputers × two scalers × seven ridge alphas '
        '`[0.1,1,10,100,1000,10000,100000]`. All imputer/scaler parameters use training data only; '
        'validation MRR selects settings. Real TRT exceeds the Gaussian and existing sentence-ridge '
        'comparators, but it trails **both structural controls on all reported recall cutoffs and MRR**. '
        'Real-minus-count/duration MRR is −0.065026; real-minus-zero/missingness MRR is −0.054043. '
        'The large gain over the sentence ridge does not establish an EEG-content advantage.','',
        'The existing sentence ridge uses its original 1,050 statistics/Welch features from normalized '
        '200 Hz voltage and an at-most-eight-second crop, freshly fitted on the same eligible cohort. '
        'The supplied TRT features use different producer fixation windows, so this is not an alignment-only '
        'ablation. [Pilot methods and full grids](../spectral_trt_pilot_v1_20261030.md) and '
        '[archived pilot evidence](spectral_trt_pilot_v1_20261030/README.md) retain all details.','',
        '### Incremental information: identical structural inputs, validation only','',
        'All three incremental conditions use the **same 62 structural inputs and frozen training-fitted '
        'structural imputer/scaler**. The combined conditions append only 840 real or matched Gaussian '
        'amplitude means. Structural-only chooses among 28 settings. Each combined model searches the '
        '**same 196 settings**: two EEG imputers × two EEG scalers × seven alphas × seven post-scaling '
        'EEG weights `[0,0.01,0.03,0.1,0.3,1,3]`. Applying weights after scaling prevents the scaler from '
        'cancelling them. Zero EEG weight reproduces the structural-only path exactly at each alpha. '
        'Zero wins ties; all three selected alpha 10 and both combined models selected EEG weight 0.01.','',
        *rows_table(inc),'',
        '| Selected validation difference | ΔRecall@1 (pp) | ΔRecall@5 (pp) | ΔRecall@10 (pp) | ΔMRR |',
        '|---|---:|---:|---:|---:|']
    for d in diffs:section.append(f"| {d['comparison']} | {100*d['recall@1']:+.4f} | {100*d['recall@5']:+.4f} | {100*d['recall@10']:+.4f} | {d['mrr']:+.6f} |")
    section+=['',
        'Real TRT has a **tiny selected aggregate validation MRR increase** over both models, but loses '
        'two Recall@1 hits. Reader-level differences contradict a consistent benefit:','',
        '| Validation reader | Queries | Real − structural MRR | Real − Gaussian-combined MRR |',
        '|---|---:|---:|---:|']
    for subject in iby['structural_real']['per_subject']:
        n=iby['structural_real']['per_subject'][subject]['n_queries']
        section.append(f"| {subject} | {n} | {diffs[0]['per_subject'][subject]['mrr']:+.6f} | {diffs[1]['per_subject'][subject]['mrr']:+.6f} |")
    section+=['',
        '**These configurations do not demonstrate a consistent useful incremental EEG advantage.** '
        'The structural baseline is strong; that finding must remain separate from the absence of a '
        'consistent additional contribution from real TRT. Structural-only here reproduces the pilot '
        'count/duration baseline and is **not an independent replication**. [Incremental methods and all '
        'settings](../trt_incremental_v1_20261030.md) and [the incremental archive](trt_incremental_v1_20261030/README.md) '
        'preserve predictions and preprocessing provenance.','',
        '### Repeated validation use and stopping status','',
        'The pilot selected six models from 28 settings each. The incremental study selected structural '
        'preprocessing from 28 settings and each combined model from **196 settings**. This gives combined '
        'models more search opportunity than structural-only, while their real/Gaussian searches match. '
        'This validation split was also used by earlier CNN, pooling and spectral investigations. '
        '**Repeated validation use and maximization over settings create selection optimism.** The '
        'scores are exploratory selection results, not independent held-out estimates, significance '
        'tests or evidence from replicated cohorts. No confidence interval based on independent words '
        'or repeated uses of these readers is justified.','',
        'Source TRT values are Hilbert-envelope amplitude summaries, not squared power or waveforms. '
        'Stored duration controls use **sample counts at 500 Hz**, not milliseconds. Source gaze alignment, '
        'full-session preprocessing, the documented gamma2 edge discrepancy and sampled fixation boundary '
        'tails remain limitations. All new learned preprocessing uses training data only. Word text and '
        'sentence IDs remain outside the model-input columns; the proposed punctuation repair remains '
        'unapplied.','',
        '**Further tuning and test evaluation are stopped.** No new fitting, tuning, feature extraction '
        'or model evaluation ran during this consolidation. The original 79 reports, transfer grids, '
        'standalone studies, feature caches and checkpoints remain preserved.','',
        'Recorded targeted software checks were **7 passed** for the spectral pilot and **4 passed** '
        'for incremental ridge. They are separate targeted checks, not a new cumulative full-suite '
        'count. The historical 139-passed/two-optional-skip record elsewhere in this report remains '
        'historical. Consolidation performs only evidence/package/render checks.','',
        '[Consolidated validation evidence](spectral_validation_consolidation_20261030/stored_evidence.json) '
        'retains full precision, the common observation/candidate lists and source-report provenance.','']
    report=ROOT/'docs/research_results/REPORT.md';text=report.read_text(encoding='utf-8')
    if '## Validation-only TRT spectral and incremental-information experiments' in text:raise ValueError('Section already present')
    text=text.replace('Draft research report, 2026-10-29. Measurements come from the user\'s Windows\nRTX 3050 execution.',
        'Research consolidation, 2026-10-30. Measurements come from local Windows execution\n(GPU neural fits and CPU ridge fits).')
    anchor='## Completion checklist'
    text=text.replace(anchor,'\n'.join(section)+'\n'+anchor,1)
    main_add=('\nThe validation-only TRT studies make a separate distinction: structural counts and\n'
        'durations retrieve sentence classes at MRR 0.292012, while real TRT adds only\n'
        '+0.000299 over structure and +0.000154 over the Gaussian-combined model. Reader\n'
        'effects change sign and Recall@1 declines. These selected, repeatedly reused\n'
        'validation scores do not demonstrate a consistent incremental EEG advantage.\n'
        'Further tuning and test evaluation are stopped.\n')
    text=text.replace('\nThese negative findings are scientifically reportable.',main_add+'\nThese negative findings are scientifically reportable.',1)
    text=text.replace('Updated 20-slide review deck and PDF','Updated 26-slide review deck and PDF')
    text=text.replace('[20-slide presentation draft]','[26-slide presentation draft]')
    text=text.replace('| Published-work discussion |','| Validation-only spectral and incremental studies | Completed on unchanged seed-42 train/validation cohort; structural baseline reported separately; no consistent incremental EEG advantage; test features unextracted |\n| Published-work discussion |',1)
    report.write_text(text,encoding='utf-8')
    # Work from the archived source package. No presentation dependency can
    # accidentally import an experiment runner or rerun a fitting command.
    source=BEFORE/'NeuroText_consolidated.pptx'
    with ZipFile(source) as z:parts={name:z.read(name) for name in z.namelist()}
    for i in range(1,21):
        root=E.fromstring(parts[f'ppt/slides/slide{i}.xml'])
        for node in root.findall('.//a:t',NS):
            if node.text and '29 OCT 2026' in node.text:node.text=node.text.replace('29 OCT 2026','30 OCT 2026')
        if i==1:
            for obj in root.findall('.//p:sp',NS):
                if '18 wider-grid exploratory transfer evaluations;' in ''.join(obj.itertext()):
                    replace_text(obj,'Waveform and transfer findings with validation-only TRT follow-ups.',16)
        parts[f'ppt/slides/slide{i}.xml']=xml(root)
    subtitle='VALIDATION ONLY  /  ZuCo1 seed 42  /  512 queries  /  298 candidates'
    pilot_rows=[['Condition','Val R@1','Val R@5','Val R@10','Val MRR']]
    labels={'real_trt':'Real TRT','matched_gaussian':'Matched Gaussian','class_shuffled':'Class shuffled','count_duration_only':'Count / duration','zero_missingness_only':'Zero / missingness','existing_sentence_ridge':'Sentence ridge'}
    for r in pilot:
        m=r['metrics'];pilot_rows.append([labels[r['condition']],f"{100*m['recall@1']:.3f}%",f"{100*m['recall@5']:.3f}%",f"{100*m['recall@10']:.3f}%",f"{m['mrr']:.6f}"])
    inc_rows=[['Model','Val R@1','Val R@5','Val R@10','Val MRR','EEG weight']]
    for r,label in zip(inc,['Structure','Structure + real','Structure + Gaussian']):
        m=r['metrics'];inc_rows.append([label,f"{100*m['recall@1']:.3f}%",f"{100*m['recall@5']:.3f}%",f"{100*m['recall@10']:.3f}%",f"{m['mrr']:.6f}",r['selection']['eeg_weight']])
    reader_rows=[['Reader','Queries','Val real - structure MRR','Val real - Gaussian MRR']]
    for p in iby['structural_real']['per_subject']:
        reader_rows.append([p.split(':')[1],iby['structural_real']['per_subject'][p]['n_queries'],f"{diffs[0]['per_subject'][p]['mrr']:+.6f}",f"{diffs[1]['per_subject'][p]['mrr']:+.6f}"])
    specs=[
        dict(title='Validation-only spectral follow-up',body='TRT pools eight 105-channel word vectors into sentence features.\nThe same 2,245 training and 512 validation observations remain.\nImputation and scaling parameters use training data only.\nCounts and missingness require explicit structural controls.',
            callout='Gaze-assisted source features retain timing and preprocessing limitations.',notes='Source: results/spectral_trt_pilot_v1_20261030/plan.json and feature_view.json. TRT values are Hilbert-envelope amplitudes. Durations are sample counts at 500 Hz. No word text or sentence identity input. Unchanged targets, no punctuation repair. No new exclusions. One all-missing training observation retained. Test features not extracted. Reused validation cohort, not a new held-out test.'),
        dict(title='Structural controls outperform real TRT',table=pilot_rows,widths=[3.45,2.05,2.05,2.05,2.2],size=19,
            callout='Count / duration validation MRR 0.292012 uses no measured TRT amplitudes.',notes='Source: results/spectral_trt_pilot_v1_20261030/validation_summary.json. All table scores are validation only, not test. Each condition selected one of 28 settings (two imputers, two scalers, seven alphas). The structural 62-feature count/duration baseline has a train-derived count basis. Real-minus-count MRR -0.065026; real-minus-zero/missingness -0.054043. All six models share observations and 298 candidates. Large gains above old sentence features do not establish EEG information.'),
        dict(title='Incremental models share all structural inputs',body='All three models share 62 structural features and one frozen transform.\nCombined models append 840 real or matched Gaussian TRT values.\nBoth combined models search 196 settings, including zero EEG weight.\nPost-scaling weights preserve an exact structural-only candidate.',
            callout='196 settings = 2 EEG imputers x 2 EEG scalers x 7 alphas x 7 weights',notes='Source: results/trt_incremental_v1_20261030/plan.json and structural_preprocessing_frozen.json. Identical raw and transformed structural blocks; training-fit structural imputation/scaling selected using structural-only validation and then frozen. Structural-only searches 28 settings. EEG weights [0,.01,.03,.1,.3,1,3] after scaling. Alphas [.1,1,10,100,1000,10000,100000]. Ascending weight wins ties. Validation MRR selects. Both combined searches have identical 196 setting tuples. Repeated use of validation across investigations creates selection optimism.'),
        dict(title='Real TRT adds a tiny validation MRR increase',table=inc_rows,widths=[3.05,1.65,1.65,1.65,2.0,1.8],size=18,
            callout='Val delta MRR: real - structure +0.000299, real - Gaussian +0.000154',notes='Source: results/trt_incremental_v1_20261030/validation_summary.json and differences.json. Validation-only selected models, 2245 training observations, 512 validation queries, 298 candidates. All alpha10. Both selected EEG weight .01, not zero, but gains are tiny. Structure median/std, real EEG median/std, Gaussian EEG mean/robust IQR. Zero EEG candidates exactly reproduce structural path. These scores are selection results after 196 settings, not independent test estimates.'),
        dict(title='Readers show opposite incremental MRR changes',table=reader_rows,widths=[1.25,1.3,4.25,4.3],size=20,
            callout='Real TRT loses two Recall@1 hits overall despite the small pooled MRR increase.',notes='Source: results/trt_incremental_v1_20261030/differences.json. Validation-only per-reader changes: ZJS216 real-structure -.003530047, real-Gaussian -.004364868. ZKW296 +.003092387 and +.003452321. Same two readers and one previously reused split, not independent cohorts. Do not pool these as replicated neural effects.'),
        dict(title='No consistent incremental EEG advantage',body='The structural baseline retrieves sentence classes from reading structure.\nReal TRT adds no consistent benefit across both validation readers.\nThe 196-setting search and repeated validation use limit inference.\nFurther tuning and test evaluation are stopped.',
            callout='Every new score is validation-only. No test features were extracted.',notes='Sources: docs/research_results/REPORT.md and the spectral/incremental archives. Separate structural retrieval from incremental EEG information. Full observation/candidate identity and hashes preserved. No confidence interval or significance claim from word observations or repeated validation use. Prior experiment files, target cache, datasets and checkpoints unchanged. No new fitting, tuning or model evaluation during consolidation. Structural baseline appears in both studies and is not an independent repeat. Historical waveform and transfer findings remain separate.')]
    types=E.fromstring(parts['[Content_Types].xml']);presentation=E.fromstring(parts['ppt/presentation.xml']);rels=E.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    ids=presentation.find('p:sldIdLst',NS);maxid=max(int(n.get('id')) for n in ids)
    maxrid=max(int(n.get('Id')[3:]) for n in rels if n.get('Id').startswith('rId'))
    for number,spec in enumerate(specs,21):
        root=E.fromstring(parts['ppt/slides/slide20.xml'])
        replace_text(shape(root,3),spec['title'],30);replace_text(shape(root,4),subtitle,16);replace_text(shape(root,6),str(number),10)
        replace_text(shape(root,9),spec['callout'],16)
        if 'table' in spec:add_table(root,spec['table'],spec['widths'],spec['size'])
        else:replace_text(shape(root,7),spec['body'],22)
        parts[f'ppt/slides/slide{number}.xml']=xml(root)
        sr=E.fromstring(parts['ppt/slides/_rels/slide20.xml.rels'])
        for rel in sr:
            if rel.get('Type').endswith('/notesSlide'):rel.set('Target',f'../notesSlides/notesSlide{number}.xml')
        parts[f'ppt/slides/_rels/slide{number}.xml.rels']=xml(sr)
        parts[f'ppt/notesSlides/notesSlide{number}.xml']=notes(parts['ppt/notesSlides/notesSlide20.xml'],spec['notes'])
        nr=E.fromstring(parts['ppt/notesSlides/_rels/notesSlide20.xml.rels'])
        for rel in nr:
            if rel.get('Type').endswith('/slide'):rel.set('Target',f'../slides/slide{number}.xml')
        parts[f'ppt/notesSlides/_rels/notesSlide{number}.xml.rels']=xml(nr)
        rid=f'rId{maxrid+number-20}'
        E.SubElement(rels,f'{{{PKG}}}Relationship',Id=rid,Type=f'{R}/slide',Target=f'slides/slide{number}.xml')
        E.SubElement(ids,f'{{{P}}}sldId',id=str(maxid+number-20),attrib={f'{{{R}}}id':rid})
        for stem,kind in [('slides/slide','slide'),('notesSlides/notesSlide','notesSlide')]:
            E.SubElement(types,f'{{{CT}}}Override',PartName=f'/ppt/{stem}{number}.xml',ContentType=f'application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml')
    parts['ppt/presentation.xml']=xml(presentation);parts['ppt/_rels/presentation.xml.rels']=xml(rels);parts['[Content_Types].xml']=xml(types)
    app=E.fromstring(parts['docProps/app.xml'])
    for element in app:
        if element.tag.rsplit('}',1)[-1] in ('Slides','Notes'):element.text='26'
    parts['docProps/app.xml']=xml(app)
    target=BUILD/'NeuroText_spectral_draft.pptx'
    if target.exists():raise FileExistsError(target)
    with ZipFile(target,'w',ZIP_DEFLATED) as z:
        for name,value in parts.items():z.writestr(name,value)
    (evidence/'README.md').write_text('# Validation-only spectral consolidation\n\n'
        'Main report and 26-slide presentation consolidate existing spectral and incremental evidence. '
        'All nine selected-model rows reuse one seed-42 validation cohort; the repeated structural '
        'baseline is not an independent replication. `stored_evidence.json` preserves full precision '
        'and observation/candidate provenance. Standalone experiment archives remain unchanged. '
        'No tuning, fitting, feature extraction or model evaluation ran during consolidation.\n',encoding='utf-8')
    print('Updated report and created 26-slide draft '+str(target))


if __name__=='__main__':main()
