"""Bounded source-prefix corroboration, not reconstruction of recording times."""
import json
from pathlib import Path
import h5py
import numpy as np
from eeg_decoding.data import mat5
from audit_word_eeg import names


def read_v5_first(path):
    def read_cells(element,endian):
        _,shape,_=mat5._header(element.payload,endian);cells=[]
        for _ in range(int(np.prod(shape))):
            e=mat5._Element(element.payload,endian);cells.append(mat5._selected(e,endian,False,mat5.MAX_FIELD_BYTES));e.finish()
        return cells
    with path.open('rb') as f:
        header=mat5._exact(f,128);endian={b'IM':'<',b'MI':'>'}[header[126:128]]
        outer=mat5._Element(f,endian);stream=mat5._Inflater(outer.payload) if outer.kind==mat5.COMPRESSED else None
        matrix=mat5._Element(stream,endian) if stream else outer
        _,_,_=mat5._header(matrix.payload,endian);fields=names(matrix.payload,endian)
        sentence=None;words=[]
        for field in fields:
            e=mat5._Element(matrix.payload,endian)
            if field=='rawData':sentence=mat5._selected(e,endian,False,mat5.MAX_FIELD_BYTES)
            elif field=='word':
                flags,shape,_=mat5._header(e.payload,endian)
                if flags&255!=2:raise ValueError('Expected word struct')
                wf=names(e.payload,endian)
                for _ in range(int(np.prod(shape))):
                    word=[]
                    for k in wf:
                        child=mat5._Element(e.payload,endian)
                        if k=='rawEEG' and child.size:
                            # rawEEG is a cell array when this word has fixations;
                            # inspect flags before dispatching an empty numeric [].
                            import io
                            payload=child.read_bytes(mat5.MAX_FIELD_BYTES)
                            import struct
                            wrapped=mat5._Element(io.BytesIO(struct.pack(endian+'II',mat5.MATRIX,len(payload))+payload+b'\0'*((-len(payload))%8)),endian)
                            flag,_,_=mat5._header(wrapped.payload,endian)
                            if flag&255==1:
                                wrapped=mat5._Element(io.BytesIO(struct.pack(endian+'II',mat5.MATRIX,len(payload))+payload+b'\0'*((-len(payload))%8)),endian)
                                word=read_cells(wrapped,endian)
                        child.finish()
                    words.append(word)
            e.finish()
        return sentence,words


def read_hdf_first(path):
    with h5py.File(path,'r') as f:
        s=f['sentenceData'];sentence=f[s['rawData'][0,0]][()].T
        w=f[s['word'][0,0]];words=[]
        for ref in w['rawEEG'][()].flatten():
            cell=f[ref]
            words.append([f[r][()].T for r in cell[()].flatten()] if h5py.check_dtype(ref=cell.dtype) is not None else [])
        return sentence,words


def exact_offsets(sentence,fixation):
    if fixation.ndim!=2 or fixation.shape[0]!=sentence.shape[0] or not np.isfinite(fixation).all():return []
    n=fixation.shape[1]
    return [int(start) for start in np.flatnonzero(sentence[0]==fixation[0,0])
            if start+n<=sentence.shape[1] and np.array_equal(sentence[:,start:start+n],fixation)]


if __name__=='__main__':
    output=Path('results/word_eeg_feasibility_20261030/containment_examples.json')
    if output.exists():raise FileExistsError(output)
    results=[]
    for dataset,subject in [('zuco1','ZAB'),('zuco2','YAC')]:
        path=Path(f'data/raw/{dataset}/results{subject}_NR.mat')
        sentence,words=read_hdf_first(path) if h5py.is_hdf5(path) else read_v5_first(path)
        count=0
        for wi,cells in enumerate(words):
            for fi,fix in enumerate(cells):
                if fix.ndim!=2 or fix.shape[0]!=105 or not np.isfinite(fix).all() or fix.shape[1]<2:continue
                results.append({'dataset':dataset,'subject':subject,'sentence_id':'0','word_index_one_based':wi+1,
                    'fixation_cell_one_based':fi+1,'sentence_shape':list(sentence.shape),'fixation_shape':list(fix.shape),
                    'exact_sentence_relative_start_samples_zero_based':exact_offsets(sentence,fix),
                    'scope':'Exact stored waveform containment only; no original acquisition timestamp inferred'})
                count+=1
                if count==5:break
            if count==5:break
    output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))
