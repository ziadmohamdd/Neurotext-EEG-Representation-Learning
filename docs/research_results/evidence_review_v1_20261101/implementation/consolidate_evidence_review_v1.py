"""Consolidate saved JSON/CSV evidence only. No ML imports, arrays or inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/research_results/evidence_review_v1_20261101'
MAIN=ROOT/'docs/research_results/REPORT.md'
METRICS=('recall@1','recall@5','recall@10','mrr')
STUDIES={'word_order':'word_order_pilot_v1_20261031','amplitude':'amplitude_audit_v1_20261101',
         'gru':'gru_pilot_v1_20261101','gru_ce':'gru_retrieval_loss_v1_20261101',
         'word_pretrain':'word_pretrain_v1_20261101','spectral':'spectral_trt_pilot_v1_20261030',
         'incremental':'trt_incremental_v1_20261030','pooling':'temporal_pooling_20261030'}
SOURCES={}
CHECKS=[]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(p):
    p=Path(p);SOURCES[str(p.relative_to(ROOT))]=sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))


def saved(study,name): return read(ROOT/'results'/STUDIES[study]/name)


def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def close(a,b,label):
    if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12):raise AssertionError((label,a,b))
    CHECKS.append(label)


def metrics_from_ranks(values):
    r=[math.inf if v is None else v for v in values]
    return {**{'recall@'+str(k):sum(v<=k for v in r)/len(r) for k in (1,5,10)},
            'mrr':statistics.fmean(1/v for v in r),'n_queries':len(r)}


def verify_rank_row(row,label):
    if 'ranks' in row:
        computed=metrics_from_ranks(row['ranks']);reported=row.get('metrics',row)
        for m in (*METRICS,'n_queries'):close(reported[m],computed[m],label+':'+m)


def prepare():
    if (OUT/'preservation_before.json').exists():raise FileExistsError('Review already prepared')
    manifest=read(OUT/'archive_manifest.json')
    for row in manifest:
        assert sha(ROOT/row['source'])==sha(ROOT/row['archive'])==row['sha256']
    protected=[p for d in ('results','checkpoints','docs','src','scripts','configs','tests') for p in (ROOT/d).rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(OUT)]
    protected += [p for p in ROOT.iterdir() if p.is_file()]
    # Targets/manifests are hashed; other dataset files are checked through metadata.
    old=read(ROOT/'results/word_order_pilot_v1_20261031/plan.json')
    protected += [Path(old['manifest']),Path(old['text_cache'])]
    write(OUT/'preservation_before.json',{str(p.relative_to(ROOT)):sha(p) for p in protected})
    write(OUT/'dataset_metadata_before.json',{str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns]
          for p in (ROOT/'data').rglob('*') if p.is_file()})
    (OUT/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),encoding='utf-8')
    print('Prepared byte-preservation inventory and verified five archival copies')


def collect():
    cv=[];outer=[];hist=[];words=[];transfer=[]
    def add(table,study,arm,metric,per,source,**extra):
        table.append({'study':study,'arm':arm,'metrics':{m:metric[m] for m in METRICS},
                      'per_reader':per,'source_json':source,**extra})
    for study in ('gru','gru_ce','word_pretrain'):
        summary=saved(study,'training_cv_summary.json');raw=saved(study,'training_cv_all_scores.json')
        epoch=None if study=='word_pretrain' else saved(study,'frozen_selection.json')['common_epoch']
        for r in raw:verify_rank_row(r,study+':'+str(r.get('reader'))+':'+str(r.get('condition',r.get('arm')))+':'+str(r.get('epoch')))
        for arm,s in summary.items():
            rows=[r for r in raw if r.get('condition',r.get('arm'))==arm and
                  (study=='word_pretrain' or r['epoch']==(0 if arm=='structural_only' else epoch))]
            assert len(rows)==8
            for m in METRICS:
                close(s['macro'][m],statistics.fmean(r[m] for r in rows),study+':macro:'+arm+':'+m)
            for r in rows:
                for m in METRICS:close(s['per_reader'][r['reader']][m],r[m],study+':reader:'+arm+':'+r['reader']+':'+m)
            add(cv,study,arm,s['macro'],s['per_reader'],f'results/{STUDIES[study]}/training_cv_summary.json',selection_epoch=epoch)
        # Check supplied delta artifacts against the underlying summaries.
        real='pretrained_real' if study=='word_pretrain' else 'real_eeg'
        for arm,gap in saved(study,'real_minus_controls.json').items():
            for m in METRICS:close(gap['macro_difference'][m],summary[real]['macro'][m]-summary[arm]['macro'][m],study+':delta:'+arm+':'+m)
            for reader,delta in gap['per_reader_difference'].items():
                for m in METRICS:close(delta[m],summary[real]['per_reader'][reader][m]-summary[arm]['per_reader'][reader][m],study+':reader-delta:'+arm+':'+reader+':'+m)
        access=saved(study,'access_summary.json')
        assert not access.get('outer_evaluated',False) and not access.get('test_evaluated',False)
    raw=saved('amplitude','inner_training_scores.json')
    for r in raw:verify_rank_row(r,'amplitude:'+r['head']+':'+r['condition']+':'+r['reader'])
    assert len(raw)==576
    for row in saved('amplitude','training_summary.json'):
        chosen=row['selection'];matches=[r for r in raw if all(r[k]==row[k] for k in ('head','condition')) and r['alpha']==chosen['alpha'] and r['weight']==chosen['weight']]
        assert len(matches)==8
        metric={m:chosen['macro_'+m] for m in METRICS}
        for m in METRICS:close(metric[m],statistics.fmean(r[m] for r in matches),'amplitude:macro:'+row['head']+':'+row['condition']+':'+m)
        add(cv,'amplitude/'+row['head'],row['condition'],metric,row['per_reader'],f"results/{STUDIES['amplitude']}/training_summary.json",selection=chosen)
    raw=saved('word_order','inner_training_scores.json');selection=saved('word_order','frozen_selection.json')
    for arm,s in selection.items():
        rows=[r for r in raw if r['condition']==arm and r['alpha']==s['alpha']];assert len(rows)==8
        metric={m:statistics.fmean(r[m] for r in rows) for m in METRICS}
        grid=next(r for r in s['grid'] if r['alpha']==s['alpha'])
        close(metric['recall@1'],grid['macro_reader_recall@1'],'ordered:CV:'+arm)
        add(cv,'word_order',arm,metric,{r['reader']:{m:r[m] for m in METRICS} for r in rows},f"results/{STUDIES['word_order']}/inner_training_scores.json",alpha=s['alpha'])
    for study in ('spectral','incremental','word_order','pooling'):
        rows=saved(study,'validation_summary.json')
        for r in rows:
            verify_rank_row(r,study+':outer:'+r.get('condition',str((r.get('pooling'),r.get('control')))))
            per=r.get('per_subject',r.get('subject_wise',{}))
            # Some early spectral summaries use no saved ranks; metrics remain source JSON facts.
            if 'ranks' in r and 'observation_identities' in r:
                for reader,m in per.items():
                    short=reader.split(':')[-1]
                    rr=[rank for rank,ident in zip(r['ranks'],r['observation_identities']) if ident[1]==short]
                    for k in METRICS:close(m[k],metrics_from_ranks(rr)[k],study+':outer-reader:'+reader+':'+k)
            add(outer,study,r.get('condition',str(r.get('pooling'))+'/'+str(r.get('control'))),r['metrics'],per,
                f'results/{STUDIES[study]}/validation_summary.json',n_queries=r['metrics']['n_queries'],candidates=len(r['candidate_keys']))
    # All 79 historical CSV rows must match actual saved JSON metrics, not just a copied CSV.
    csvpath=ROOT/'docs/research_results/retrieval.csv';SOURCES[str(csvpath.relative_to(ROOT))]=sha(csvpath)
    csvrows=list(csv.DictReader(csvpath.open(encoding='utf-8-sig')));assert len(csvrows)==79
    for row in csvrows:
        path=ROOT/row['file'];j=read(path)
        for m in (*METRICS,'n_queries'):close(row[m],j['metrics'][m],'historical CSV:'+row['file']+':'+m)
        assert row['candidate_pool_sha256']==j['candidate_pool_sha256']
        per=j.get('subject_wise',{})
        if per and all(m in v for v in per.values() for m in (*METRICS,'n_queries')):
            for m in METRICS:close(j['metrics'][m],sum(v[m]*v['n_queries'] for v in per.values())/sum(v['n_queries'] for v in per.values()),'historical weighted subjects:'+row['file']+':'+m)
        add(hist,'historical_export',path.stem,j['metrics'],per,str(path.relative_to(ROOT)),
            protocol=j['protocol'],control=j['training_control'],datasets=j['test_datasets'],seed=j['seed'],
            evaluation_scope='historical_validation_diagnostic' if path.stem=='zuco1_ZJS_validation' else 'historical_test',
            candidate_pool=j.get('candidate_pool'),candidates=len(j['candidate_keys']),n_queries=j['metrics']['n_queries'])
    # Original report tables use these named CSVs; cross-check all their values too.
    for name in ('cnn_controls_summary.csv','ssl_controls_summary.csv','labram_controls_summary.csv'):
        p=ROOT/'docs/research_results'/name;SOURCES[str(p.relative_to(ROOT))]=sha(p)
        for row in csv.DictReader(p.open(encoding='utf-8-sig')):
            paths=row.get('file',row.get('source_files')).split(';')
            for path in paths:
                j=read(ROOT/path)
                for m in (*METRICS,'n_queries'):close(row[m],j['metrics'][m],'old summary:'+name+':'+path+':'+m)
    # CNN candidate comparison adds six historical test evaluations, stored separately.
    csummary=read(ROOT/'results/cnn_candidate_comparison_20261030/summary.json')
    for r in csummary:
        matches=list((ROOT/'results/cnn_candidate_comparison_20261030').glob('**/evaluation.json'))
        candidates=[]
        for path in matches:
            j=read(path)
            if j.get('training_control')==r['control'] and all(math.isclose(j['metrics'][m],r[m],abs_tol=1e-12) for m in METRICS):candidates.append((path,j))
        assert candidates,'CNN comparison lacks matching source evaluation'
        path,j=candidates[0]
        add(hist,'cnn_candidates/'+r['scope'],r['control'],r,j.get('subject_wise',{}),str(path.relative_to(ROOT)),
            candidates=r['n_candidates'],n_queries=r['n_queries'],seed=42,protocol='subject',datasets=['zuco1'],control=r['control'],evaluation_scope='historical_test')
    for folder in sorted((ROOT/'results/word_pretrain_v1_20261101').glob('cv_*')):
        for p in folder.glob('pretrained_*/word_holdout.json'):
            j=read(p);metric=metrics_from_ranks(j['ranks'])
            for m in METRICS:close(j['all_word_slots'][m],metric[m],'word:'+folder.name+':'+p.parent.name+':'+m)
            assert len(j['query_target_indices'])==len(j['ranks']) and sum(v<0 for v in j['query_target_indices'])==j['oov_slots']
            words.append({'arm':p.parent.name,'reader':folder.name[3:],'metrics':j['all_word_slots'],
                          'observed_metrics':j['observed_eeg_slots'],'oov_slots':j['oov_slots'],
                          'vocabulary_size':j['vocabulary_size'],'source_json':str(p.relative_to(ROOT))})
    wider=read(ROOT/'docs/research_results/wider_grid_transfer/summary.json')
    for r in wider['runs']:
        path=Path(r['report']);j=read(path)
        close(r['recall_at_1'],j['metrics']['recall@1'],'wider:source:R1')
        close(r['mrr'],j['metrics']['mrr'],'wider:source:MRR')
        add(transfer,r['direction'],r['control'],j['metrics'],j.get('subject_wise',{}),str(path.relative_to(ROOT)),
            seed=r['seed'],candidates=r['n_candidates'],n_queries=r['n_queries'],identity_audited=r['identity_audited'])
    return cv,outer,hist,words,transfer


def export_tables(groups):
    for name,rows in groups.items():
        flat=[]
        for row in rows:
            item={k:v for k,v in row.items() if k not in ('metrics','per_reader','selection') and not isinstance(v,(dict,list))}
            flat.append({**item,**row['metrics']})
        columns=list(dict.fromkeys(k for r in flat for k in r))
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(flat)


def compare(rows,study,real,controls=None):
    by={r['arm']:r for r in rows if r['study']==study};a=by[real];results=[]
    for arm,b in by.items():
        if arm==real or controls is not None and arm not in controls:continue
        shared=set(a['per_reader'])&set(b['per_reader'])
        per={s:{m:a['per_reader'][s][m]-b['per_reader'][s][m] for m in METRICS} for s in sorted(shared)}
        results.append({'study':study,'real':real,'control':arm,'difference':{m:a['metrics'][m]-b['metrics'][m] for m in METRICS},
                        'positive_recall1_readers':sum(d['recall@1']>1e-12 for d in per.values()),
                        'positive_mrr_readers':sum(d['mrr']>1e-12 for d in per.values()),'reader_count':len(per),'per_reader_difference':per})
    return results


def metric_table(rows,label):
    lines=[f'| {label} | R@1 % | R@5 % | R@10 % | MRR |','|---|---:|---:|---:|---:|']
    for name,metric in rows:
        lines.append('| '+name+' | '+' | '.join(f'{100*metric[m]:.4f}' if m!='mrr' else f'{metric[m]:.6f}' for m in METRICS)+' |')
    return lines


def audit_experiment_readmes(cv,outer,words):
    """Compare printed metric cells with saved JSON, respecting display precision."""
    passed=[];unmatched=[]
    for study in ('word_order','amplitude','gru','gru_ce','word_pretrain'):
        p=ROOT/'docs/research_results'/STUDIES[study]/'README.md';heading='';headers=None
        SOURCES[str(p.relative_to(ROOT))]=sha(p)
        for number,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
            if line.startswith('#'):heading=line;headers=None
            if not line.startswith('|'):
                headers=None
                continue
            cells=[x.strip().strip('*') for x in line.strip('|').split('|')]
            if any('R@1' in x or 'Recall@1' in x for x in cells):headers=cells;continue
            if not headers or all(re.fullmatch(r'[-: ]+',c) for c in cells):continue
            columns={}
            for i,h in enumerate(headers):
                for m in METRICS:
                    label=m.replace('recall','r')
                    if re.search(re.escape(label.lower())+r'(?!\d)',h.lower().replace('recall','r')):columns[m]=i
            if not columns:continue
            is_delta=any('Δ' in h or 'Real −' in h for h in headers)
            def lookup(st,arm):return next((r for r in cv+outer if r['study']==st and r['arm']==arm),None)
            expected=None;metric=None;reader=None
            if study=='amplitude':
                head=cells[0];arm=cells[1] if len(cells)>1 else ''
                expected=lookup('amplitude/'+head,arm)
                if expected:
                    reader=next((c for c in cells if c in expected['per_reader']),None)
                    metric=expected['per_reader'][reader] if reader else expected['metrics']
                elif is_delta and head in ('semantic_embedding','known_sentence_class'):
                    metric={m:0. for m in METRICS}
            elif 'Uniform' in cells[0]:
                n=298 if study=='word_order' else 300
                metric={**{'recall@'+str(k):k/n for k in (1,5,10)},'mrr':sum(1/i for i in range(1,n+1))/n}
            elif study=='gru_ce' and cells[0].startswith('Cosine'):
                metric=lookup('gru','real_eeg')['metrics']
            elif study=='gru_ce' and cells[0].startswith('Full-candidate'):
                metric=lookup('gru_ce','real_eeg')['metrics']
            elif study=='word_pretrain' and 'Word task' in heading:
                matches=[r for r in words if r['arm']==cells[0]]
                if matches:metric={m:statistics.fmean(r['metrics'][m] for r in matches) for m in METRICS}
            else:
                arm=cells[0].replace('ordered_real_minus_','')
                st=study
                source=outer if study=='word_order' else cv
                expected=next((r for r in source if r['study']==st and r['arm']==arm),None)
                if expected:
                    reader=next((c for c in cells[1:3] if c in expected['per_reader']),None)
                    metric=expected['per_reader'][reader] if reader else expected['metrics']
                    if is_delta:
                        real='ordered_real' if study=='word_order' else ('pretrained_real' if study=='word_pretrain' else 'real_eeg')
                        a=next(r for r in source if r['study']==study and r['arm']==real)
                        metric={m:a['metrics'][m]-metric[m] for m in METRICS}
            if metric is None:
                unmatched.append({'file':str(p.relative_to(ROOT)),'line':number,'cells':cells});continue
            for m,index in columns.items():
                value=cells[index].rstrip('%').replace('−','-')
                try: actual=float(value)
                except ValueError:continue
                target=metric[m]*(1 if m=='mrr' else 100)
                decimals=len(value.split('.')[-1]) if '.' in value else 0
                assert abs(actual-target)<=.50001*10**(-decimals),(p,number,m,actual,target)
                passed.append({'file':str(p.relative_to(ROOT)),'line':number,'metric':m})
    write(OUT/'existing_readme_metric_checks.json',{'verified_cells':len(passed),'unmatched_metric_rows':unmatched,'cells':passed})
    if unmatched:raise AssertionError(('Unmapped existing metric rows',unmatched))
    return len(passed)


def build(refresh=False):
    if (OUT/'verification.json').exists() and not refresh:raise FileExistsError('Completed review exists; preserve it unless refreshing this review explicitly')
    cv,outer,hist,words,transfer=collect()
    cohort=saved('word_order','plan.json')
    assert len(cohort['inner_readers'])==8 and len(cohort['train_observations'])==2245
    assert len(cohort['inner_candidate_keys'])==300 and len(cohort['validation_observations'])==512 and len(cohort['candidate_keys'])==298
    for st in ('gru','gru_ce'):
        assert saved(st,'frozen_selection.json')['common_epoch']==8
        assert saved(st,'plan.json')['configuration']['epoch_candidates']==[4,8,12]
    wp=saved('word_pretrain','plan.json')['configuration']
    assert wp['word_steps']==128 and wp['sentence_steps']==256
    assert saved('word_pretrain','feasibility.json')['feasible']
    for row in cv:
        if row['study'].startswith('amplitude/') and row['arm'] in ('structural_real','structural_gaussian'):
            assert row['selection']['weight']==0
    printed_cells=audit_experiment_readmes(cv,outer,words)
    diagnostic_path=ROOT/'results/metrics/zuco1_ZJS_validation.json'
    diagnostic=read(diagnostic_path)
    assert 'metrics' not in diagnostic
    diagnostic_note={'source_json':str(diagnostic_path.relative_to(ROOT)),'sha256':sha(diagnostic_path),
        'artifact_type':'Manifest/data validation summary, not a retrieval evaluation','retrieval_metrics_present':False}
    write(OUT/'nonperformance_artifact.json',diagnostic_note)
    # Remove only an empty CSV created by this review's earlier artifact iteration.
    empty_csv=OUT/'historical_validation_diagnostic.csv'
    if empty_csv.exists():
        assert empty_csv.stat().st_size<=2
        empty_csv.unlink()
    export_tables({'training_cv':cv,'outer_validation':outer,'historical_export':hist,
        'historical_test':[r for r in hist if r['evaluation_scope']=='historical_test'],
        'word_task':words,'exploratory_transfer':transfer})
    gaps=[]
    for study,real in [('gru','real_eeg'),('gru_ce','real_eeg'),('word_pretrain','pretrained_real'),('word_order','ordered_real')]:
        gaps+=compare(cv,study,real)
    for head in ('semantic_embedding','known_sentence_class'):
        gaps+=compare(cv,'amplitude/'+head,'structural_real',('structural_only','structural_gaussian'))
        gaps+=compare(cv,'amplitude/'+head,'eeg_only',('gaussian_only',))
    for study,real in [('spectral','real_trt'),('incremental','structural_real'),('word_order','ordered_real'),('pooling','four_segment/real')]:
        for g in compare(outer,study,real):g['study']='outer/'+g['study'];gaps.append(g)
    with (OUT/'participant_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','study','arm','reader',*METRICS]);writer.writeheader()
        for scope,rows in [('training_cv',cv),('outer_validation',outer),('historical_export',hist),('exploratory_transfer',transfer)]:
            for r in rows:
                for reader,m in r['per_reader'].items():
                    writer.writerow({'scope':scope,'study':r['study'],'arm':r['arm'],'reader':reader,**{k:m[k] for k in METRICS}})
    word_summary=[]
    for arm in dict.fromkeys(r['arm'] for r in words):
        matching=[r for r in words if r['arm']==arm]
        word_summary.append({'arm':arm,'metrics':{m:statistics.fmean(r['metrics'][m] for r in matching) for m in METRICS},
            'per_reader':{r['reader']:r['metrics'] for r in matching}})
    write(OUT/'word_task_macro.json',word_summary)
    # Scope-specific historical selections: original JSONs, never pooled with CV/validation.
    historical=[];historical_groups=[]
    for family,file in [('CNN','cnn_controls_summary.csv'),('SSL','ssl_controls_summary.csv'),('LaBraM','labram_controls_summary.csv')]:
        for r in csv.DictReader((ROOT/'docs/research_results'/file).open(encoding='utf-8-sig')):
            historical.append((family+'/'+r['dataset']+'/'+r['condition'],{m:float(r[m]) for m in METRICS}))
            path=ROOT/r.get('file',r.get('source_files')).split(';')[0];j=read(path)
            historical_groups.append({'study':'historical/'+family+'/'+r['dataset'],'arm':r['condition'],
                'metrics':j['metrics'],'per_reader':j['subject_wise'],'pool_hash':j['candidate_pool_sha256']})
    for study in sorted({r['study'] for r in historical_groups}):
        group=[r for r in historical_groups if r['study']==study]
        assert len({r['pool_hash'] for r in group})==1
        gaps+=compare(group,study,'real')
    for study in sorted({r['study'] for r in hist if r['study'].startswith('cnn_candidates/')}):
        group=[r for r in hist if r['study']==study]
        for g in compare(group,study,'real'):g['study']='historical/'+g['study'];gaps.append(g)
    write(OUT/'control_differences.json',gaps)
    word_group=[dict(r,study='word_task') for r in word_summary]
    word_gaps=compare(word_group,'word_task','pretrained_real')
    write(OUT/'word_control_differences.json',word_gaps)
    for dataset in ('zuco1','zuco2'):
        for control in ('real','gaussian','shuffled'):
            matching=[r for r in hist if r['study']=='historical_export' and r['protocol']=='subject_sentence' and r['datasets']==[dataset] and r['control']==control]
            assert sorted(r['seed'] for r in matching)==[42,43,44]
            historical.append(('Unseen-content seed mean/'+dataset+'/'+control,
                {m:statistics.fmean(r['metrics'][m] for r in matching) for m in METRICS}))
    for r in hist:
        if r['study'].startswith('cnn_candidates/'):
            historical.append((r['study']+'/'+r['arm'],r['metrics']))
    transfer_mean=[]
    transfer_differences=[]
    for direction in sorted({r['study'] for r in transfer}):
        for arm in ('real','gaussian','shuffled'):
            matching=[r for r in transfer if r['study']==direction and r['arm']==arm];assert len(matching)==3
            transfer_mean.append((direction+'/'+arm,{m:statistics.fmean(r['metrics'][m] for r in matching) for m in METRICS}))
        for seed in (42,43,44):
            a=next(r for r in transfer if r['study']==direction and r['seed']==seed and r['arm']=='real')
            b=next(r for r in transfer if r['study']==direction and r['seed']==seed and r['arm']=='gaussian')
            transfer_differences.append({'direction':direction,'seed':seed,'difference':{m:a['metrics'][m]-b['metrics'][m] for m in METRICS},
                'per_reader_difference':{s:{m:a['per_reader'][s][m]-b['per_reader'][s][m] for m in METRICS} for s in a['per_reader']}})
    write(OUT/'transfer_differences.json',transfer_differences)
    # Preserve complete source-level evidence separately from concise tables.
    write(OUT/'stored_evidence.json',{'training_cv':cv,'outer_validation':outer,'historical_export':hist,
        'nonperformance_artifact':diagnostic_note,'word_task':words,'exploratory_transfer':transfer})
    rel='evidence_review_v1_20261101/'
    lines=['# EEG research review: completed experiments and evidence', '',
        'Updated 2026-11-01 from saved results only. **The tested configurations do not demonstrate a useful, consistent EEG contribution beyond the strongest matched controls.** This does not establish that EEG contains no language information.', '',
        'No training, tuning, feature extraction, checkpoint inference, outer-validation evaluation or test evaluation ran during this consolidation. Numerical checks summarize previously saved ranks and metrics. The [pre-update report and presentation archive]('+rel+'archive_manifest.json) contains byte-identical copies. The existing PowerPoint/PDF remain unchanged historical deliverables; this report includes later studies they do not cover.', '',
        '## What changed and what was measured', '',
        '| Experiment | Change and controls | Evidence-supported outcome |', '|---|---|---|',
        '| Historical statistical/CNN/SSL/LaBraM | Waveform statistics, supervised convolution, reconstruction pretraining and frozen adapted pretrained features; real, Gaussian and class-shuffled supervision controls | No consistent useful real-EEG advantage. Memorizing a small training subset verifies optimization capacity, not reader generalization. |',
        '| CNN candidate loss | Fixed CNN and cohort; batch candidates versus full training-candidate loss, each with real/Gaussian/shuffled arms | Both real arms retain the same Recall@1; broader candidates do not establish a useful gain. Separate historical test comparison. |',
        '| Temporal pooling | Global versus four source-time segments; matched real/Gaussian/shuffled fits | Same real Recall@1; the segment variant trails its Gaussian control. Validation only. |',
        '| Spectral TRT mean | Supplied word-band amplitude means with count/mask structure; Gaussian, shuffled, structure-only, mask-only and waveform-ridge comparators | Better than a waveform baseline is not an isolated EEG increment: both non-amplitude controls are stronger. |',
        '| Incremental TRT | Same structural block plus real or Gaussian amplitudes; post-scaling EEG weights include zero | Tiny selected MRR increment, lower Recall@1, opposite reader effects. |',
        '| Ordered-word ridge | Training-fitted PCA and four source-word-position bins versus mean; Gaussian, order-shuffled, class-shuffled, mask and structure controls | One extra outer-validation hit versus mean; Gaussian, order-shuffled and structure remain stronger. |',
        '| Relative-amplitude/head audit | Full-width signed-log relative channel features averaged by sentence; semantic embedding versus known-class head; Gaussian, shuffled, structure and combined arms | Structure dominates; selected combined models use zero EEG weight for both heads. |',
        '| Full-sequence GRU | Small projection, one GRU layer, embedding head; mean, Gaussian, order-shuffled, class-shuffled, mask and structure controls | Small selected advantage over Gaussian, inconsistent across readers and far below structure; training-only gate fails. |',
        '| GRU retrieval loss | Same inputs/architecture/budget/init; positive-only cosine versus full-candidate cross entropy | Prediction concentration decreases, but sentence retrieval does not improve. |',
        '| Word pretraining | Exact exported-word class targets, projection-only transfer; fresh same-sentence and equal-update baselines, Gaussian/shuffled/mask stages and structure | No useful sentence-retrieval gain over fresh training or strongest controls. Word scores are reported separately. |', '',
        '## Training-reader CV: sentence outcomes', '',
        'These are equal-reader macro development scores on the original eight training readers and 2,245 observations, with 300 fixed sentence candidates. Recall values are percentages; **MRR is not accuracy**. Heads and representations are distinct tasks/settings. These are not outer-validation or historical test scores.', '']
    lines+=metric_table([(r['study']+'/'+r['arm'],r['metrics']) for r in cv],'Study / condition')
    lines += ['', 'The amplitude known_sentence_class head is closed-set classification of repeated known sentence classes; its structural performance is not semantic EEG decoding. Repeated structural rows reuse the same observations and recipe, not independent replications.', '',
        '## Outer validation: separate reused development cohort', '',
        'These previously executed scores use 512 queries from ZJS/ZKW and 298 candidates. Spectral/incremental settings were selected on this validation set. Ordered-word ridge selected alphas within training readers before its one outer evaluation, but the outer readers already had extensive development use. The later amplitude, GRU, GRU-loss and word-pretraining pilots did not evaluate outer validation.', '']
    lines+=metric_table([(r['study']+'/'+r['arm'],r['metrics']) for r in outer],'Study / condition')
    lines += ['', '## Historical test: previously saved scores only', '',
        'CNN/SSL use 592 ZuCo1 queries and 984 ZuCo2 queries, with 300/344 candidates. LaBraM uses 592/983 queries and 300/344 candidates on a separately adapted channel/cohort recipe. Unseen-content rows below are equal-seed means across overlapping splits with only 60/68 candidates; their scores are not comparable to full-pool retrieval. The separate CNN-candidate comparison uses 592 queries and 300 candidates. None of these old test sets is a new independent confirmation or a current model-selection target.', '']
    lines+=metric_table(historical,'Historical method / dataset / condition')
    lines += ['', 'The complete original export and separate CNN-loss outcomes are retained in [historical_export.csv]('+rel+'historical_export.csv), with original JSON paths, seeds and pool sizes. The separately stored ZJS_validation JSON is a manifest/data audit without retrieval scores; it is [classified separately]('+rel+'nonperformance_artifact.json) and excluded from performance tables. Duplicate timestamped LaBraM reports have equal saved values and are displayed once, not treated as replications. Statistical-ridge robustness and the adapted statistical comparator remain in the complete export; they are not silently discarded.', '',
        '## Exploratory cross-release transfer: a fourth scope', '',
        'Transfer evaluates the full target release, not an isolated held-out target test cohort. The following are equal-seed means of the saved wider-grid runs. Source-validation scores select ridge regularization. Cross-release participant independence is unresolved; inclusive pools retain shared source/target texts. The original-grid seed-42 runs and wider-grid runs are separate experiments and must not be pooled as repeats.', '']
    lines+=metric_table(transfer_mean,'Direction / condition (seed mean)')
    lines += ['', 'Forward real-minus-Gaussian MRR is negative at every saved seed. Reverse differences change sign. Seed-level and participant-level differences are in [transfer_differences.json]('+rel+'transfer_differences.json). Repeated seeds reuse target readers and texts, so they do not establish independent cohorts.', '',
        '## Real-minus-control differences and reader consistency', '',
        'Recall differences are percentage points; MRR differences remain reciprocal-rank units. Positive-reader counts describe paired differences on the same participants and are not significance tests. Full differences against every comparator and per-reader values are in [control_differences.json]('+rel+'control_differences.json) and [participant_metrics.csv]('+rel+'participant_metrics.csv).', '',
        '| Scope / real arm minus comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Positive MRR readers |',
        '|---|---:|---:|---:|---:|---:|---:|']
    selected=[]
    for g in gaps:
        if (g['study'] in ('gru','gru_ce') and g['control'] in ('matched_gaussian','structural_only')) or \
           (g['study']=='word_pretrain' and g['control'] in ('fresh_equal_steps','fresh_same_sentence','pretrained_gaussian','pretrained_missingness','structural_only')) or \
           (g['study'].startswith('amplitude/') and g['real']=='structural_real') or \
           (g['study']=='outer/incremental' and g['control'] in ('structural_only','structural_gaussian')) or \
           (g['study']=='outer/word_order' and g['control'] in ('mean_real','matched_gaussian','structural_only')):
            selected.append(g)
    for g in selected:
        d=g['difference'];lines.append('| '+g['study']+': '+g['real']+' - '+g['control']+' | '+
            ' | '.join(f'{100*d[m]:+.4f}' if m!='mrr' else f'{d[m]:+.6f}' for m in METRICS)+
            f" | {g['positive_recall1_readers']}/{g['reader_count']} | {g['positive_mrr_readers']}/{g['reader_count']} |")
    inc=[r for r in outer if r['study']=='incremental'];by={r['arm']:r for r in inc}
    lines += ['', 'The incremental-validation real-minus-structure MRR effects are '+
        ', '.join(f"{s}: {by['structural_real']['per_reader'][s]['mrr']-by['structural_only']['per_reader'][s]['mrr']:+.6f}" for s in by['structural_real']['per_reader'])+
        '. Opposite signs prevent interpreting the small aggregate gain as consistent EEG information. Structural count/duration retrieval is a separate, strong annotation-derived finding. A model using real EEG without a structural block is not a direct incremental ablation; the combined TRT analyses supply that comparison.', '',
        '## Word task: separate supervised outcome', '',
        'Exact exported labels and saved word TRT rows were paired across the declared training cohort before fitting. Text supplies target classes only. Each fold builds its vocabulary from seven fit readers, retains missing slots, and scores any out-of-vocabulary held-out target as a miss with reciprocal rank zero. No difficult sentence is excluded. These reader splits repeat sentence classes; no unseen-word generalization is established.', '']
    lines+=metric_table([(r['arm'],r['metrics']) for r in word_summary],'Word-stage condition (macro across readers)')
    lines += ['', 'These are word-class scores, not sentence accuracy. The missingness-only word model is stronger than real EEG here. Frequent lexical classes and gaze-derived missingness can support these scores without demonstrated EEG language information. Per-reader word outcomes are in [word_task.csv]('+rel+'word_task.csv).', '',
        '## Selection history and limitations', '',
        'Spectral arms searched 28 settings each on outer-validation MRR; incremental combined arms searched 196 each, versus 28 for structure. Ordered-word ridge selected among three alphas using eight training-reader folds. The amplitude audit selected among three alphas and three EEG weights for combined arms (three alphas for individual blocks), on the same readers later summarized. Both GRU objectives selected a shared epoch from 4/8/12 by real-EEG training-CV Recall@1, preferring earlier ties; every control used that epoch. Both chose epoch 8. This gives real EEG selection opportunity and makes the reported CV scores optimistic development descriptions.', '',
        'Word pretraining instead fixed 128 word updates plus 256 sentence updates with no epoch search. The same-sentence fresh baseline has 256 sentence updates; the equal-update baseline has 384 sentence updates with an optimizer reset after 128. Equal update counts do not match FLOPs or sequence exposure. Only the projection transfers; the word head and optimizer state are discarded and GRU/output-head initialization is shared.', '',
        'All later downstream scalers, Gaussian moments and vocabularies fit fold-training readers only; masks and missing-word positions remain separate from padding. However, earlier experiments already informed hypotheses and representation choices. Overlapping folds, repeated words/sentences, reused validation and historical test inspection are not independent cohorts. No aggregate final accuracy, causal explanation, independent confirmation or word-level confidence interval follows.', '',
        'The measurement remains gaze/stimulus assisted. Source-session filtering/reference histories, exact producer revisions, gamma2 definition discrepancies and possible physical boundary tails remain unresolved. Historical waveform Gaussian controls match each observation/channel standard deviation, retaining observation-specific amplitude information. Later TRT Gaussian controls use moments fitted on training words and preserve exact word counts and missingness. These are different nulls; neither preserves full real EEG covariance/distribution. Cross-release participant identities remain unverified; strict independent transfer remains unestablished. Local execution does not establish an independent cloud reproduction.', '',
        '## Verified findings versus explanations', '',
        '**Verified engineering/evidence findings:** exported text can differ from a stimulus CSV without mismatching the selected source/cache pair; the proposed punctuation repair remains unapplied. Source word/cache pairing and saved-rank reproduction passed the earlier recorded audits. Interpreter/sandbox access and a historical large-fixture storage failure were execution constraints, not evidence of bad EEG labels. The earlier report retains provenance for superseded ridge artifacts, which must not be counted as independent successful methods.', '',
        '**Observed diagnostics, not confirmed failure causes:** reader amplitude norms differ; a relative transform reduces that spread; PCA retains only part of standardized variance; positive-only GRU predictions concentrate on a few candidates; full-candidate loss spreads predictions. None proves that amplitude, PCA loss, coarse sentence supervision or objective concentration caused poor retrieval. The bounded follow-ups did not demonstrate the desired improvement. No confirmed current pairing/ranking bug was identified in these reviewed paths. Low scores do not prove incorrect pairing or absence of language information in EEG.', '',
        '## Review decision and reproducibility', '',
        'Stop experiments under the current instruction. Further work would need a revised, falsifiable hypothesis and a narrower question with controls, or better independent reader/stimulus data for a generalization claim. Reusing these development scores for another broad search would not supply independent evidence. There is no authorization here for more fitting, target repair, feature extraction, evaluation or correspondence.', '',
        '[Supervisor-review note]('+rel+'SUPERVISOR_NOTE.md). [Numerical/artifact checks]('+rel+'verification.json). [Source JSON hashes]('+rel+'source_hashes.json). [Byte-preservation check]('+rel+'preservation_after.json). The earlier report and presentation are archived, and all previous experiment files remain intact. Historical software-test counts in the archived report are historical; this consolidation runs artifact and numerical checks only.', '']
    historical_gaps=[g for g in gaps if g['study'].startswith('historical/')]
    extra=['', '### Historical test controls: participant consistency', '',
        'These paired differences use the same saved test queries/candidate pools within each comparison. They do not supply new test evaluations or independent replications.', '',
        '| Historical real minus control | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Positive MRR readers |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in historical_gaps:
        d=g['difference'];extra.append('| '+g['study']+' - '+g['control']+' | '+
            ' | '.join(f'{100*d[m]:+.4f}' if m!='mrr' else f'{d[m]:+.6f}' for m in METRICS)+
            f" | {g['positive_recall1_readers']}/{g['reader_count']} | {g['positive_mrr_readers']}/{g['reader_count']} |")
    word_extra=['', 'Word real-minus-control differences are also saved separately in [word_control_differences.json]('+rel+'word_control_differences.json).', '',
        '| Word-stage real minus comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers |',
        '|---|---:|---:|---:|---:|---:|']
    for g in word_gaps:
        d=g['difference'];word_extra.append('| '+g['control']+' | '+
            ' | '.join(f'{100*d[m]:+.4f}' if m!='mrr' else f'{d[m]:+.6f}' for m in METRICS)+
            f" | {g['positive_recall1_readers']}/{g['reader_count']} |")
    report='\n'.join(lines).replace('## Word task: separate supervised outcome','\n'.join(extra)+'\n\n## Word task: separate supervised outcome')
    report=report.replace('## Selection history and limitations','\n'.join(word_extra)+'\n\n## Selection history and limitations')
    # Write only after all source checks passed and archive identity was rechecked.
    for r in read(OUT/'archive_manifest.json'):assert sha(ROOT/r['archive'])==r['sha256']
    MAIN.write_text(report,encoding='utf-8')
    by_cv={(r['study'],r['arm']):r for r in cv};by_outer={(r['study'],r['arm']):r for r in outer}
    short=['# Concise research review — 2026-11-01', '',
        '**The tested configurations do not demonstrate a useful, consistent EEG contribution beyond the strongest matched controls.** This does not show that EEG contains no language information. This review checks saved artifacts only; it launches no experiment or evaluation.', '',
        'The [updated main report](../REPORT.md) contains the complete scope-separated tables, method inventory, control differences and selection history. [Supervisor note](SUPERVISOR_NOTE.md). [Byte-identical pre-update report/presentation archive](archive_manifest.json). The presentation remains unchanged and does not include the later pilots.', '',
        '## Changes, controls and outcomes', '',
        '| Study | Change and matched checks | Supported finding |','|---|---|---|',
        '| Waveform/statistical/CNN/SSL/LaBraM | Learned or frozen representations; Gaussian and shuffled controls | No consistent useful EEG advantage across the matched comparisons. |',
        '| CNN candidate loss / temporal pooling | Full training candidates or source-time bins; real/Gaussian/shuffled fits | Loss change does not improve real top-1; pooling does not beat its Gaussian control. |',
        '| Spectral / incremental TRT | Word amplitude means, then a fixed structural block plus real or Gaussian EEG | Structure is strong; adding EEG gives tiny selected MRR changes and inconsistent reader effects. |',
        '| Ordered-word ridge | PCA position bins versus averaging; Gaussian, order/class shuffle, mask and structure | Small mean-to-order difference does not exceed stronger controls. |',
        '| Relative-amplitude/head audit | Full-width relative amplitudes; semantic versus known-class targets | Both combined heads select zero EEG weight. |',
        '| GRU / full-candidate GRU loss | Full ordered sequences, then loss-only change; matched initialization and controls | Prediction concentration falls with full-candidate loss, but retrieval does not improve. |',
        '| Word pretraining | Exact exported labels as targets; projection transfer, fresh baselines and matched two-stage controls | No useful sentence gain over fresh training or strongest controls. |', '',
        '## Training-reader CV', '',
        'Selected equal-reader development scores from the original eight training readers; later word pretraining uses a fixed final checkpoint. Recall is percent. **MRR is not accuracy.** These are not outer-validation or test results. Complete conditions: [training_cv.csv](training_cv.csv).', '']
    short_cv=[('amplitude/semantic_embedding','eeg_only'),('amplitude/semantic_embedding','gaussian_only'),
        ('amplitude/semantic_embedding','structural_only'),('amplitude/known_sentence_class','structural_only'),
        ('gru','real_eeg'),('gru','matched_gaussian'),('gru_ce','real_eeg'),('gru_ce','missingness_only'),
        ('word_pretrain','pretrained_real'),('word_pretrain','fresh_same_sentence'),('word_pretrain','fresh_equal_steps'),
        ('word_pretrain','pretrained_gaussian'),('word_pretrain','pretrained_missingness')]
    short+=metric_table([(s+'/'+a,by_cv[s,a]['metrics']) for s,a in short_cv],'Study / condition')
    short += ['', '## Outer validation', '',
        'Previously executed results on 512 reused validation queries and 298 candidates. No new outer evaluation occurred. The later amplitude/GRU/loss/word-pretraining pilots have no outer results. Complete conditions: [outer_validation.csv](outer_validation.csv).', '']
    short_outer=[('spectral','real_trt'),('spectral','matched_gaussian'),('spectral','count_duration_only'),
        ('incremental','structural_real'),('incremental','structural_only'),('incremental','structural_gaussian'),
        ('word_order','ordered_real'),('word_order','matched_gaussian'),('word_order','structural_only')]
    short+=metric_table([(s+'/'+a,by_outer[s,a]['metrics']) for s,a in short_outer],'Study / condition')
    short += ['', '## Historical test', '',
        'Selected old test comparisons below retain their original cohorts and candidate pools; no test inference ran. The full report includes SSL, CNN-loss, unseen-content and other historical conditions. Different cohorts/pools prevent one final-accuracy ranking. The separately stored ZJS_validation file is a manifest audit without retrieval metrics, excluded from performance tables.', '']
    short+=metric_table([(name,m) for name,m in historical if name.startswith(('CNN/','LaBraM/')) and not name.endswith('/shuffled')],
                        'Historical method / dataset / condition')
    short += ['', 'Cross-release transfer is a separate exploratory scope, not an independent target-test cohort. Forward real-minus-Gaussian MRR is negative across the saved seeds; reverse differences change sign. Participant overlap between releases remains unresolved. [Transfer metrics](exploratory_transfer.csv).', '',
        '## Controls, structure and reader consistency', '',
        'Strong structural retrieval uses annotation-derived count/duration information; it does not establish incremental EEG information. Both amplitude combined heads select zero EEG weight. Incremental validation gains in aggregate MRR have opposite signs for the two readers and lower Recall@1.', '',
        '| Real condition minus comparator | ΔR@1 pp | ΔMRR | Readers with positive R@1 gap |',
        '|---|---:|---:|---:|']
    for study,control in [('gru','matched_gaussian'),('gru','structural_only'),('gru_ce','matched_gaussian'),
                           ('word_pretrain','fresh_same_sentence'),('word_pretrain','fresh_equal_steps'),
                           ('word_pretrain','pretrained_gaussian'),('word_pretrain','pretrained_missingness')]:
        g=next(g for g in gaps if g['study']==study and g['control']==control);d=g['difference']
        short.append(f"| {study} - {control} | {100*d['recall@1']:+.4f} | {d['mrr']:+.6f} | {g['positive_recall1_readers']}/{g['reader_count']} |")
    short += ['', 'Full Recall@1/5/10 and MRR differences, including historical controls and individual readers: [control_differences.json](control_differences.json), [participant_metrics.csv](participant_metrics.csv). Positive counts are descriptive, not significance tests.', '',
        '## Word task, separately', '',
        'Fold-training exact-string classes; text is target metadata only. Missing words remain, and out-of-vocabulary targets count as misses. These are word scores, not sentence accuracy. Reader folds repeat sentence classes; no unseen-word claim follows.', '']
    short+=metric_table([(r['arm'],r['metrics']) for r in word_summary],'Word-stage condition')
    short += ['', '## Interpretation and remaining uncertainty', '',
        'Spectral/incremental settings were selected on reused outer validation. Ordered ridge and amplitude settings used training-reader folds; both GRU objectives selected epoch 8 from 4/8/12 on real-EEG CV. Word pretraining fixed its stages and supplied both same-sentence and equal-update fresh baselines; optimizer-update matching is not FLOP matching. Repeated development use, overlapping folds and repeated words/sentences limit every claim. These are not independent cohorts.', '',
        'No confirmed current pairing/ranking bug explains the negative scores. Earlier source/cache and saved-rank checks passed. Exported-text versus stimulus-CSV differences and execution-environment problems do not establish EEG mispairing. Amplitude differences, retained PCA variance and prediction concentration are observations, not proven causes. The tested follow-ups did not establish the proposed improvement.', '',
        'Source gaze-assisted segmentation, upstream session processing and physical boundary uncertainties remain. Gaussian controls are limited null models: historical waveform nulls retain observation-specific amplitude, while TRT nulls preserve masks/counts with training-word moments. Better independent reader/stimulus data are needed for generalization claims. Any future experiment needs a revised falsifiable hypothesis or narrower question, not another broad search on this development history. Work stops here.', '',
        '[Verification](verification.json) reports saved-result arithmetic and artifact checks, not a new model reproduction. [Preservation](preservation_after.json) confirms prior files and dataset metadata, with only the authorized main-report edit and this newly created builder changing during consolidation.', '']
    (OUT/'README.md').write_text('\n'.join(short),encoding='utf-8')
    note='''# Supervisor review: EEG project

The project implemented local data preparation, fixed-pool sentence retrieval,
classical and neural models, reconstruction pretraining, an adapted frozen EEG
backbone, matched negative controls, reader-disjoint folds, training-only
preprocessing and versioned experiment provenance. Source/cache pairing,
padding/missingness handling, target preservation and saved-rank consistency
were checked. This is verified engineering and a reportable controlled negative
result; it is not a validated EEG-to-language decoder.

The tested waveform, spectral, ordered-word, relative-amplitude, GRU,
full-candidate-loss and word-pretraining configurations did not establish a
useful, consistent EEG contribution beyond the strongest matched controls.
Structural count/duration features retrieve repeated sentence classes much more
strongly; that result must remain separate from incremental EEG information.
Word-task accuracy is also separate from sentence retrieval.

Uncertainty remains about reader/stimulus generalization, source recording
preprocessing and physical boundaries, cross-release participant independence,
and which representations or supervision might preserve useful EEG information.
No confirmed current pairing/ranking bug explains the negative results. The
data do not establish that EEG contains no language information.

Further experiments would need a revised falsifiable hypothesis and a narrower
research question with explicit controls. Generalization claims also need
independent readers/stimuli and an untouched evaluation plan. Repeated tuning
on the existing development history cannot provide that confirmation. No new
experiment, evaluation or correspondence is authorized by this review.

See [the verified research review](README.md) and its scope-separated tables.
'''
    (OUT/'SUPERVISOR_NOTE.md').write_text(note,encoding='utf-8')
    # Verify every generated metric table line survives formatting/write exactly.
    for table_rows,label in [([(r['study']+'/'+r['arm'],r['metrics']) for r in cv],'Study / condition'),
                             ([(r['study']+'/'+r['arm'],r['metrics']) for r in outer],'Study / condition'),
                             (historical,'Historical method / dataset / condition'),(transfer_mean,'Direction / condition (seed mean)'),
                             ([(r['arm'],r['metrics']) for r in word_summary],'Word-stage condition (macro across readers)')]:
        for line in metric_table(table_rows,label):assert line in MAIN.read_text(encoding='utf-8')
    preservation=finish_preservation()
    verification={'passed':True,'checks_scope':'Saved JSON/CSV/rank arithmetic and file artifacts only; no model execution',
        'numeric_equality_checks':len(CHECKS),'existing_experiment_readme_metric_cells_checked':printed_cells,
        'training_cv_rows':len(cv),'outer_validation_rows':len(outer),'historical_export_rows':len(hist),
        'word_reader_rows':len(words),'exploratory_transfer_rows':len(transfer),
        'original_historical_csv_rows_checked_against_JSON':79,'source_files_hashed':len(SOURCES),
        'all_new_metric_tables_generated_from_verified_JSON':True,'local_markdown_links_valid':True,
        'presentation_unchanged':preservation['presentation_unchanged'],
        'limits':'Stored arithmetic/provenance consistency is not independent scientific reproduction or raw-data validation.'}
    # Make the verification link target exist before final link verification on repeat inspection.
    write(OUT/'verification.json',verification)
    write(OUT/'source_hashes.json',SOURCES)
    write(OUT/'commands.json',{'prepare':'.\\.venv\\Scripts\\python.exe scripts/consolidate_evidence_review_v1.py --prepare',
        'consolidate':'.\\.venv\\Scripts\\python.exe scripts/consolidate_evidence_review_v1.py',
        'artifact_refresh':'.\\.venv\\Scripts\\python.exe scripts/consolidate_evidence_review_v1.py --refresh',
        'execution':'Existing .venv; Python standard library only; base interpreter requires access outside sandbox',
        'prohibited_actions_executed':[]})
    for p in (MAIN,OUT/'README.md',OUT/'SUPERVISOR_NOTE.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' not in link:assert (p.parent/link).exists(),(p,link)
    (OUT/'implementation').mkdir(exist_ok=True);shutil.copyfile(Path(__file__),OUT/'implementation'/Path(__file__).name)
    print(json.dumps(verification,indent=2))


def finish_preservation():
    before=read(OUT/'preservation_before.json');changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    normalized={p.replace('\\','/') for p in changed}
    assert normalized<= {'docs/research_results/REPORT.md','scripts/consolidate_evidence_review_v1.py'} and 'docs/research_results/REPORT.md' in normalized,changed
    inv=read(OUT/'dataset_metadata_before.json');altered=[p for p,s in inv.items() if [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=s]
    assert not altered
    archives=read(OUT/'archive_manifest.json')
    for r in archives:
        assert sha(ROOT/r['archive'])==r['sha256']
        if r['source']!='docs/research_results/REPORT.md':assert sha(ROOT/r['source'])==r['sha256']
    result={'passed':True,'protected_files':len(before),'unchanged_protected_files':len(before)-len(changed),
        'authorized_modified_files':changed,'dataset_metadata_files_unchanged':len(inv),'byte_identical_archival_copies':len(archives),
        'presentation_unchanged':True,'no_training_tuning_feature_extraction_or_model_evaluation':True}
    write(OUT/'preservation_after.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:build(args.refresh)
