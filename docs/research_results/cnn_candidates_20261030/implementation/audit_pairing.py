"""Read-only audit of source sentence slots and the unchanged ZuCo1 CNN cohort."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from eeg_decoding.data.dataset import read_manifest,load_signal,sentence_key
from eeg_decoding.data.mat5 import iter_sentence_fields
from eeg_decoding.data.metadata import sha256_file,read_eeg_metadata
from eeg_decoding.preprocessing.filtering import filter_resample
from eeg_decoding.utils.logging import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a fresh audit folder')
    output.mkdir(parents=True)
    qc=read_manifest('data/manifests/zuco1_duration_qc.jsonl')
    all_records=read_manifest('data/processed/zuco1_batch/combined.jsonl')
    qc_ids={(r.subject,r.sentence_id) for r in qc}
    by_subject=defaultdict(dict)
    for r in all_records: by_subject[r.subject][int(r.sentence_id)]=r
    subjects=[]; failures=[]; flat=Counter(); hashes=defaultdict(list); durations=[]; peaks=[]
    source_text_by_id=defaultdict(set); cropped=0; examples=[]
    for subject,records in sorted(by_subject.items()):
        first=next(iter(records.values()))
        raw=Path(first.provenance['source'])
        stored_meta=json.loads(Path(f'data/processed/zuco1_batch/{subject}/metadata.json').read_text())
        schema=json.loads(Path(f'data/processed/zuco1_batch/{subject}/schema.json').read_text())
        meta=read_eeg_metadata(stored_meta[0]['source'])
        source_hash=sha256_file(raw)
        count=0; missing=[]; text_matches=0; signal_matches=0; maximum_error=0.; retained=0
        for index,signal,text in iter_sentence_fields(raw,schema['signal_field'],schema['text_field']):
            count+=1
            source_text_by_id[index].add(sentence_key(text))
            if index not in records:
                missing.append(dict(index=index,missing_eeg=bool(signal.size==0 or (signal.size==1 and np.isnan(signal).all()))))
                continue
            r=records[index]
            good=(r.text==text and r.sentence_id==str(index) and r.provenance['matlab_sentence_index']==index+1)
            text_matches+=int(good)
            if not good: failures.append(f'{subject}/{index}: text/index mismatch')
            if source_hash!=r.provenance['source_sha256']: failures.append(f'{subject}/{index}: source checksum changed')
            if r.channels!=meta['channels'] or schema['sfreq']!=meta['sfreq']:
                failures.append(f'{subject}/{index}: channel/rate mismatch')
            expected=filter_resample(signal,meta['sfreq'],target_sfreq=r.sfreq,already_filtered=True)
            actual=load_signal(r)
            same=actual.shape==expected.shape and np.array_equal(actual,expected)
            signal_matches+=int(same)
            if actual.shape==expected.shape:
                maximum_error=max(maximum_error,float(np.abs(actual-expected).max()))
            if not same: failures.append(f'{subject}/{index}: resampled waveform mismatch')
            recipe=r.provenance['preprocessing']
            if recipe.get('bandpass_applied') or recipe.get('notch_applied') or not recipe['anti_alias_resampling']:
                failures.append(f'{subject}/{index}: unexpected preprocessing')
            if (subject,str(index)) not in qc_ids: continue
            retained+=1
            durations.append(actual.shape[1]/r.sfreq); peaks.append(float(np.abs(actual).max()))
            cropped+=actual.shape[1]>1600
            flat.update(label for label,std in zip(r.channels,actual.std(1)) if std<1e-6)
            hashes[hashlib.sha256(actual.tobytes()).hexdigest()].append(f'{subject}/{index}')
            if index in [min(records),sorted(records)[len(records)//2],max(records)]:
                examples.append(dict(subject=subject,zero_based_index=index,matlab_index=index+1,text=text,
                                     raw_shape=list(signal.shape),processed_shape=list(actual.shape)))
        excluded=[dict(sentence_id=str(i),duration_seconds=load_signal(r).shape[1]/r.sfreq)
                  for i,r in records.items() if (subject,str(i)) not in qc_ids]
        if any(e['duration_seconds']>.1 for e in excluded): failures.append(subject+': unexpected duration exclusion')
        if any(not m['missing_eeg'] for m in missing): failures.append(subject+': skipped slot not missing EEG')
        entry=dict(subject=subject,source=str(raw),source_sha256=source_hash,source_slots=count,
                   prepared=len(records),duration_qc_retained=retained,missing_slots=missing,
                   text_and_indices_matched=text_matches,resampled_arrays_bitwise_matched=signal_matches,
                   maximum_absolute_resampling_difference=maximum_error,duration_exclusions=excluded,
                   metadata=meta,stored_metadata_hash_matches=meta['sha256']==stored_meta[0]['sha256'])
        if not entry['stored_metadata_hash_matches']: failures.append(subject+': metadata source changed')
        subjects.append(entry)
        write_json(output/'progress.json',subjects)
        print(f'{subject}: {text_matches}/{len(records)} paired slots, {signal_matches} exact waveform reconstructions; retained={retained}',flush=True)
    conflicts={str(i):sorted(keys) for i,keys in source_text_by_id.items() if len(keys)>1}
    if conflicts: failures.append('Canonical text differs across subjects for a sentence slot')
    report=dict(status='passed' if not failures else 'failed',dataset='zuco1',source_subjects=subjects,
                cohort_observations=len(qc),prepared_observations=len(all_records),
                text_slot_conflicts=conflicts,failures=failures,examples=examples,
                duplicate_signal_groups=[v for v in hashes.values() if len(v)>1],
                flat_channel_observation_counts=dict(flat),cropped_at_8_seconds_count=int(cropped),
                retained_duration_quantiles_seconds=np.quantile(durations,[0,.25,.5,.75,1]).tolist(),
                retained_absolute_peak_quantiles_uv=np.quantile(peaks,[0,.5,.95,1]).tolist(),
                confirmed_scope='Source export rawData/content slot identity and exact cached resampling, not independent sentence-clock verification',
                unresolved=['Original export segmentation/trigger clock not independently reconstructed',
                            'One header per subject cannot establish all run-specific filter/reference histories',
                            'Microvolt units are schema/source-documented, not independently calibrated',
                            'Paper-level processing does not prove the exact processing of each rawData export'])
    write_json(output/'audit.json',report)
    if failures: raise RuntimeError(f'Audit found {len(failures)} failures; review before training')


if __name__=='__main__': main()
