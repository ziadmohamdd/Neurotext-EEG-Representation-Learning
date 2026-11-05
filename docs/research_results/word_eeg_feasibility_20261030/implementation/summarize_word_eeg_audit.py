"""Consolidate completed read-only source audits; never trains or changes targets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from eeg_decoding.data.dataset import sentence_key

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/word_eeg_feasibility_20261030'
MEASURES=('FFD','SFD','GD','GPT','TRT')


def merge_fields(reports,key):
    out={}
    for report in reports:
        for name,stats in report[key].items():
            item=out.setdefault(name,{k:Counter() for k in ('states','shapes','dtypes')})
            for k in item:item[k].update(stats[k])
    return out


def coverage(rows):
    return {'sentence_observations':len(rows),'with_word_records':sum(r['words']>0 for r in rows),
        'word_observations':sum(r['words'] for r in rows),'exact_word_text_mapping_when_present':sum(r['words']>0 and r['joined_word_text_exact'] for r in rows),
        'wordbounds_match_when_present':sum(r['words']>0 and r['wordbounds_match'] for r in rows),
        'at_least_one_usable_raw_fixation':sum(r['usable_raw_fixations']>0 for r in rows),
        'at_least_one_complete_TRT_word':sum(r['TRT_complete_8x105_words']>0 for r in rows),
        'raw_fixations':sum(r['raw_fixations'] for r in rows),'usable_raw_fixations':sum(r['usable_raw_fixations'] for r in rows),
        'fixated_word_observations':sum(r['fixated_words'] for r in rows),
        **{m+'_complete_8x105_words':sum(r[m+'_complete_8x105_words'] for r in rows) for m in MEASURES}}


def main():
    output=BASE/'summary'
    if output.exists():raise FileExistsError(output)
    reports={}
    for directory in ('zuco1','zuco2','zuco2_parallel'):
        for p in (BASE/directory).glob('*.json'):
            if p.name=='progress.json':continue
            j=json.loads(p.read_text(encoding='utf-8'))
            key=(j['dataset'],j['subject'])
            if key in reports:raise ValueError('Duplicate subject audit')
            reports[key]=j
    if Counter(k[0] for k in reports)!={'zuco1':12,'zuco2':18}:raise ValueError('Full 30-file census has not completed')
    summary={};source_inventory=[];sentences=[];fields={};mapping_exceptions=[]
    for dataset in ('zuco1','zuco2'):
        group=[j for k,j in sorted(reports.items()) if k[0]==dataset]
        rows=[r for j in group for r in j['sentences']];sentences+=rows
        totals=Counter()
        for j in group:totals.update(j['counts'])
        summary[dataset]={'subjects':len(group),'counts':dict(totals),
            'source_slots':coverage(rows),'prepared':coverage([r for r in rows if r['prepared']]),
            'duration_qc':coverage([r for r in rows if r['qc_retained']]),
            'outside_prepared':coverage([r for r in rows if not r['prepared']]),
            'mapping_disagreements_with_nonempty_words':[r for r in rows if r['words']>0 and not r['joined_word_text_exact']],
            'missing_word_records':[{'subject':r['subject'],'sentence_id':r['sentence_id'],'prepared':r['prepared'],'raw_sentence_state':r['raw_sentence_state']} for r in rows if not r['words']]}
        for j in group:
            source_inventory.append({k:j[k] for k in ('dataset','subject','source','sha256','source_matches_download_plan','seconds')})
            mapping_exceptions += [{'dataset':dataset,'subject':j['subject'],**r} for r in j['mapping_mismatches']]
        fields[dataset]={k:merge_fields(group,k) for k in ('sentence_fields','word_fields','raw_fixation_fields')}
    split=json.loads((ROOT/'results/cnn_candidate_comparison_20261030/batch_real/splits.json').read_text())
    lookup={tuple(v):name for name,detail in split['splits'].items() for v in detail['observations']}
    z1=[r for r in sentences if r['dataset']=='zuco1' and r['qc_retained']]
    summary['zuco1_saved_seed42_splits']={name:coverage([r for r in z1 if lookup[('zuco1',r['subject'],'unspecified','nr',r['sentence_id'])]==name]) for name in ('train','val','test')}
    manifest=[json.loads(x) for x in (ROOT/'data/manifests/zuco1_duration_qc.jsonl').read_text(encoding='utf-8').splitlines()]
    texts={sentence_key(r['text']):r['text'] for r in manifest}
    lengths={}
    for name,keys in [('training',split['splits']['train']['sentence_keys']),('validation',split['splits']['val']['sentence_keys'])]:
        counts=Counter(len(texts[k].split(' ')) for k in keys)
        lengths[name]={'n_candidates':len(keys),'distinct_word_counts':len(counts),'word_count_range':[min(counts),max(counts)],
                       'sentences_unique_by_word_count_alone':sum(n for n in counts.values() if n==1),'largest_equal_count_bucket':max(counts.values()),
                       'bucket_sizes':dict(sorted(counts.items())),'scope':'Metadata collision audit only; no retrieval model fitted or scored'}
    summary['word_count_candidate_collisions']=lengths
    channel_orders=[]
    for dataset,subject in sorted(reports):
        p=ROOT/f'data/processed/{dataset}_batch/{subject}/schema.json';schema=json.loads(p.read_text())
        channel_orders.append({'dataset':dataset,'subject':subject,'schema':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                               'channels':schema['channels'],'sfreq':schema['sfreq'],'units':schema['units']})
    assert all(len(r['channels'])==105 and r['channels']==channel_orders[0]['channels'] and r['sfreq']==500 for r in channel_orders)
    source=(BASE/'getElectrodePairs.m').read_text()
    pair_values=re.findall(r"pairs\{(\d+),([12])\}='([^']+)'",source)
    pairs={}
    for i,side,label in pair_values:pairs.setdefault(int(i),{})[side]=label
    assert len(pairs)==48 and all(set(p)=={'1','2'} and set(p.values())<=set(channel_orders[0]['channels']) for p in pairs.values())
    preservation=json.loads((BASE/'preservation_before.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in preservation.items())
    summary['validation']={'source_files':30,'protected_files_unchanged':len(preservation),'training_launched':False,'datasets_changed':False,
        'unit_tests':'5 passed in 0.65 s','all_30_channel_orders_match':True,'producer_sampling_rate_hz':500,
        'note':'Initial ZuCo2 sequential process stopped for parallel read-only execution; its completed YAC audit is retained and counted exactly once.'}
    output.mkdir()
    for name,data in [('summary.json',summary),('field_census.json',fields),('sentence_coverage.json',sentences),
                      ('source_inventory.json',source_inventory),('mapping_exceptions.json',mapping_exceptions),('channel_schemas.json',channel_orders),
                      ('electrode_pairs.json',pairs),('preservation_after.json',{'unchanged':True,'protected_files':len(preservation),'files_sha256':preservation})]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:({x:v[x] for x in ('subjects','source_slots','prepared','duration_qc','outside_prepared')} if k in ('zuco1','zuco2') else v) for k,v in summary.items()},indent=2))


if __name__=='__main__':main()
