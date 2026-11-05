"""Read-only training-source pairing audit and frozen pilot preparation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import shutil
import subprocess
import torch
import numpy as np
from eeg_decoding.data.dataset import read_manifest
from eeg_decoding.data.word_targets_v1 import iter_labelled_training_slots, check_pairing
from run_gru_retrieval_loss import ROOT, SOURCE, sha, write, ident

CONFIG=ROOT/'configs/word_pretrain_v1_20261101.json'


def preservation_finish(output):
    before=json.loads((output/'preservation_before.json').read_text())
    inventory=json.loads((output/'dataset_metadata_before.json').read_text())
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    altered=[p for p,stat in inventory.items() if [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=stat]
    result={'unchanged':not changed and not altered,'hashed_files':len(before),'dataset_metadata_files':len(inventory),
            'changed':changed,'metadata_changed':altered}
    write(output/'preservation_after.json',result)
    if changed or altered: raise RuntimeError('Protected files changed')


def main(output):
    if output.exists(): raise FileExistsError('Fresh versioned output required')
    old=json.loads((SOURCE/'plan.json').read_text()); view=json.loads((SOURCE/'feature_view.json').read_text())
    cfg=json.loads(CONFIG.read_text())
    assert sha(SOURCE/'word_sequences.npz')==view['cache_sha256']
    assert sha(old['manifest'])==old['manifest_sha256'] and sha(old['text_cache'])==old['text_cache_sha256']
    records={tuple(ident(r)):r for r in read_manifest(old['manifest'])}
    train=[records[tuple(k)] for k in old['train_observations']]
    readers=sorted({r.subject for r in train})
    assert readers==['ZDM','ZDN','ZGW','ZJM','ZJN','ZKH','ZMG','ZPH'] and len(train)==2245
    sources={s['subject']:s for s in view['sources'] if s['subject'] in readers}
    protected=[p for d in ('results','docs','checkpoints','configs','src','scripts','tests') for p in (ROOT/d).rglob('*')
               if p.is_file() and '__pycache__' not in p.parts]
    protected += [p for p in ROOT.iterdir() if p.is_file()]
    protected += [Path(old['manifest']),Path(old['text_cache'])]
    protected += [p for p in (ROOT/'data/processed/zuco1_trt_mean_pilot_v1_20261030').glob('*') if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    inventory={str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/'data').rglob('*') if p.is_file()}
    for s in sources.values():
        p=Path(s['path']); inventory[str(p.resolve())]=[p.stat().st_size,p.stat().st_mtime_ns]
    output.mkdir(parents=True)
    write(output/'preservation_before.json',before); write(output/'dataset_metadata_before.json',inventory)
    (output/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    write(output/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','environment_upgraded':False})
    implementations=[CONFIG,Path(__file__),ROOT/'scripts/run_word_pretrain_v1.py',ROOT/'tests/test_word_pretrain_v1.py',
        ROOT/'src/eeg_decoding/data/word_targets_v1.py',ROOT/'src/eeg_decoding/data/mat5.py',ROOT/'src/eeg_decoding/data/trt_view.py',
        ROOT/'src/eeg_decoding/evaluation/word_pretrain_v1.py',ROOT/'src/eeg_decoding/evaluation/gru_pilot.py',
        ROOT/'src/eeg_decoding/evaluation/gru_retrieval_loss.py',ROOT/'src/eeg_decoding/evaluation/controls.py',
        ROOT/'src/eeg_decoding/evaluation/spectral_pilot.py',ROOT/'src/eeg_decoding/evaluation/incremental_ridge.py',
        ROOT/'scripts/run_gru_retrieval_loss.py',ROOT/'src/eeg_decoding/evaluation/retrieval.py']
    (output/'implementation').mkdir()
    for p in implementations: shutil.copyfile(p,output/'implementation'/p.name)
    plan={'configuration':cfg,'configuration_sha256':sha(CONFIG),'source_cache_sha256':view['cache_sha256'],
        'manifest':old['manifest'],'manifest_sha256':sha(old['manifest']),'text_cache':old['text_cache'],'text_cache_sha256':sha(old['text_cache']),
        'train_observations':old['train_observations'],'training_candidate_keys':old['inner_candidate_keys'],
        'candidate_pool_sha256':sha_bytes(old['inner_candidate_keys']), 'readers':readers,
        'source_limitations':old['limitations'],'scope':'Training CV only, no outer or test access under any outcome',
        'implementation_sha256':{str(p.relative_to(ROOT)):sha(p) for p in implementations},
        'authorization':'Explicit user authorization for feasibility and one bounded word-pretraining comparison'}
    write(output/'plan.json',plan)
    with np.load(SOURCE/'word_sequences.npz',allow_pickle=False) as z:
        words,offsets=z['train_words'],z['train_offsets']
    sequences=[words[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    wanted=defaultdict(dict)
    for i,r in enumerate(train): wanted[r.subject][int(r.sentence_id)]=(i,r)
    label_rows=[None]*len(train); provenance=[]
    try:
        for reader in readers:
            source=Path(sources[reader]['path'])
            digest=sha(source)
            if digest!=sources[reader]['sha256']: raise ValueError('Training source hash mismatch: '+reader)
            count=0
            for slot,parent,labels,source_words in iter_labelled_training_slots(source,wanted[reader]):
                i,r=wanted[reader][slot]
                check_pairing(parent,labels,source_words,r.text,sequences[i])
                label_rows[i]=labels; count+=1
            assert count==len(wanted[reader])
            provenance.append({'reader':reader,'path':str(source),'sha256':digest,'paired_observations':count})
            write(output/'pairing_progress.json',provenance)
            print('PAIRED',reader,count,'source/cache EEG rows and exact exported word targets',flush=True)
        assert all(row is not None for row in label_rows)
        write(output/'exported_word_targets.json',label_rows)
        write(output/'feasibility.json',{'feasible':True,'paired_observations':len(train),'word_slots':int(offsets[-1]),
            'complete_eeg_words':int(np.isfinite(words).all(1).sum()),'sources':provenance,
            'source_cache_values_equal_including_missingness':True,'joined_exported_words_equal_unchanged_parent':True,
            'word_row_offsets_equal_cache':True,'punctuation_repair_applied':False,'stimulus_csv_used':False,
            'source_fields_decoded':['sentenceData.content','word.content','word.TRT_t1,t2,a1,a2,b1,b2,g1,g2'],
            'source_scope':'Eight training-reader files; only declared training slots decoded; other slots skipped',
            'documentation':'docs/word_eeg_feasibility_20261030.md: Sentence-word and fixation mapping evidence',
            'word_label_cache_sha256':sha(output/'exported_word_targets.json')})
    except Exception as exc:
        write(output/'feasibility.json',{'feasible':False,'blocking_evidence':str(exc),'sources_completed':provenance})
        preservation_finish(output)
        raise
    preservation_finish(output)
    print('FEASIBLE; configuration frozen; no fitting or outer/test access',flush=True)


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
