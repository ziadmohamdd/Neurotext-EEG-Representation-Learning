"""Narrow, bounded Level-5 TRT reader: word text/identities are never decoded."""
import math
import struct
import numpy as np
from . import mat5

BANDS=('t1','t2','a1','a2','b1','b2','g1','g2')
SELECTED={"TRT","nFixations",*("TRT_"+b for b in BANDS)}


def field_names(stream,endian):
    _,data=mat5._small_field(stream,endian,4);width=struct.unpack(endian+'i',data)[0]
    _,data=mat5._small_field(stream,endian)
    if width<=0 or len(data)%width:raise ValueError('Malformed struct fields')
    return [data[i:i+width].split(b'\0')[0].decode() for i in range(0,len(data),width)]


def word_array(element,endian):
    if not element.size:return np.empty((0,840)),np.empty(0),np.empty(0)
    flags,shape,_=mat5._header(element.payload,endian)
    if flags&255!=2:return np.empty((0,840)),np.empty(0),np.empty(0)
    fields=field_names(element.payload,endian);vectors=[];durations=[];counts=[]
    for _ in range(math.prod(shape)):
        values={}
        for name in fields:
            child=mat5._Element(element.payload,endian)
            if name in SELECTED:values[name]=mat5._selected(child,endian,False,mat5.MAX_FIELD_BYTES)
            child.finish()  # In particular, never decode word.content/rawEEG/rawET.
        bands=[]
        for suffix in BANDS:
            a=values.get('TRT_'+suffix,np.empty(0))
            if a.size==0:bands.append(np.full(105,np.nan))
            elif a.shape in ((1,105),(105,1)):bands.append(np.asarray(a,dtype=np.float64).reshape(-1))
            else:raise ValueError(f'Unexpected TRT_{suffix} shape {a.shape}')
        vector=np.concatenate(bands)
        if np.isinf(vector).any():raise ValueError('Infinite source TRT value')
        vectors.append(vector)
        for name,output in [('TRT',durations),('nFixations',counts)]:
            a=values.get(name,np.empty(0))
            if not a.size:output.append(np.nan)
            elif a.size==1:output.append(float(a.flat[0]))
            else:raise ValueError('Expected a scalar duration/count')
    return np.asarray(vectors,dtype=np.float64).reshape(-1,840),np.asarray(durations),np.asarray(counts)


def iter_trt_sentences(path):
    """Yield original slot, parent target (pairing check only), EEG, TRT, nFix."""
    with open(path,'rb') as f:
        header=mat5._exact(f,128);endian={b'IM':'<',b'MI':'>'}.get(header[126:128])
        if endian is None:raise ValueError('TRT pilot requires audited Level-5 files')
        outer=mat5._Element(f,endian)
        compressed=mat5._Inflater(outer.payload) if outer.kind==mat5.COMPRESSED else None
        matrix=mat5._Element(compressed,endian) if compressed else outer
        flags,shape,name=mat5._header(matrix.payload,endian)
        if name!='sentenceData' or flags&255!=2:raise ValueError('Expected sentenceData struct')
        fields=field_names(matrix.payload,endian)
        for index in range(math.prod(shape)):
            text=None;words=None
            for name in fields:
                child=mat5._Element(matrix.payload,endian)
                if name=='content':text=mat5._selected(child,endian,True,mat5.MAX_FIELD_BYTES)
                elif name=='word':words=word_array(child,endian)
                child.finish()
            if text is None or words is None:raise ValueError('Missing sentence/word field')
            yield index,text,*words
        if matrix.payload.remaining:raise ValueError('Trailing struct bytes')
        matrix.finish()
        if compressed:compressed.finish();outer.finish()


def mean_features(words):
    """Preserve all-missing coordinates as NaN for later train-only imputation."""
    valid=np.isfinite(words);n=valid.sum(0)
    out=np.full(words.shape[1],np.nan,dtype=np.float64)
    np.divide(np.where(valid,words,0).sum(0),n,out=out,where=n>0)
    return out


def structure(words):
    n=len(words);complete=int(np.isfinite(words).all(1).sum())
    return np.array([n,complete,complete/n if n else 0.],dtype=np.float64)


def count_duration(words,durations,fixations):
    n,complete,fraction=structure(words)
    duration=durations[np.isfinite(durations)];fix=fixations[np.isfinite(fixations)]
    values=[n,int((fix>0).sum()),complete,n-complete,1-fraction,float(fix.sum()),
            float(fix.mean()) if len(fix) else np.nan,float(fix.std()) if len(fix) else np.nan,
            float(duration.sum()),*[float(fn(duration)) if len(duration) else np.nan for fn in (np.mean,np.std,np.min,np.median,np.max)]]
    return np.asarray(values,dtype=np.float64)
