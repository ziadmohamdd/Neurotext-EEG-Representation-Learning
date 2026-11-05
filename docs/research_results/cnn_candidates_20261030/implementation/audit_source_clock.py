"""Bounded corroboration against one available continuous source recording."""
import json
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
from eeg_decoding.data.mat5 import iter_sentence_fields


def read_value(f,node):
    values=node[()].flatten(order='F')
    if h5py.check_dtype(ref=node.dtype) is not None:
        return [read_value(f,f[v]) if v else None for v in values]
    if node.attrs.get('MATLAB_class')==b'char': return ''.join(chr(int(v)) for v in values)
    return float(values[0]) if len(values)==1 else values.tolist()


def main():
    output=Path('results/pairing_audit_20261030/continuous_recording_sample.json')
    if output.exists(): raise FileExistsError(output)
    header=Path('data/raw/zuco1/oip_ZDM_NR3_EEG.mat')
    raw=Path('data/raw/zuco1/resultsZDM_NR.mat')
    matches=[]
    with h5py.File(header,'r') as f:
        eeg=f['EEG']; data=eeg['data']; first_channel=data[:,0]
        types=read_value(f,eeg['event']['type']); latencies=read_value(f,eeg['event']['latency'])
        events=[dict(type=t,matlab_latency=l) for t,l in zip(types,latencies)]
        meta=dict(header=str(header),shape_on_disk=list(data.shape),srate=read_value(f,eeg['srate']),
                  ref_field=read_value(f,eeg['ref']),setname=read_value(f,eeg['setname']),
                  event_type_counts=dict(Counter(str(t) for t in types)),first_events=events[:20])
        for index,signal,text in iter_sentence_fields(raw):
            x=np.asarray(signal,dtype=np.float32)
            if x.ndim!=2 or x.shape[0]!=105 or x.shape[1]<2: continue
            for start in np.flatnonzero(first_channel==x[0,0]):
                end=int(start)+x.shape[1]
                if end>data.shape[0]: continue
                # Prefix check bounds the number of full-window reads.
                if not np.array_equal(data[int(start):int(start)+min(8,x.shape[1]),:].T,x[:,:8]): continue
                if np.array_equal(data[int(start):end,:].T,x):
                    near_start=[e for e in events if abs(float(e['matlab_latency'])-(int(start)+1))<=2]
                    near_end=[e for e in events if abs(float(e['matlab_latency'])-end)<=2]
                    matches.append(dict(sentence_id=str(index),matlab_sentence_index=index+1,text=text,
                        matlab_start_sample=int(start)+1,matlab_end_sample=end,n_samples=x.shape[1],
                        near_start_events=near_start,near_end_events=near_end))
                    break
    report=dict(scope='One available ZDM NR3 continuous preprocessed recording; bounded exact matching, no resampling',
                metadata=meta,exact_waveform_matches=matches,match_count=len(matches),
                limitation='A waveform match corroborates export provenance and a recording-clock interval; it does not independently map trigger codes to stimulus text or verify every recording')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(match_count=len(matches),metadata=meta),indent=2),flush=True)


if __name__=='__main__': main()
