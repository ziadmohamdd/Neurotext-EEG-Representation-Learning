"""Read-only census of nested ZuCo word EEG. No datasets, targets or models written.

Level-5 decoding is bounded to a sentence and numeric chunks; v7.3 axes are
reported in MATLAB order. Sentence rawData values are deliberately not reread.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import h5py
import numpy as np
from eeg_decoding.data import mat5

BANDS = ('t1','t2','a1','a2','b1','b2','g1','g2')
MEASURES = ('FFD','SFD','GD','GPT','TRT')


def numeric(array, shape=None, keep=True):
    a=np.asarray(array)
    result={'kind':'numeric','shape':list(a.shape if shape is None else shape),'dtype':str(a.dtype),
            'size':int(a.size),'finite':int(np.isfinite(a).sum()),'nan':int(np.isnan(a).sum()),'inf':int(np.isinf(a).sum())}
    if keep and a.size<=4096:result['data']=a.flatten(order='F').tolist()
    return result


def names(stream,endian):
    _,data=mat5._small_field(stream,endian,4);width=struct.unpack(endian+'i',data)[0]
    _,data=mat5._small_field(stream,endian)
    if width<=0 or len(data)%width:raise ValueError('Invalid struct names')
    return [data[i:i+width].split(b'\0')[0].decode() for i in range(0,len(data),width)]


def v5_node(element,endian,path=''):
    if element.kind!=mat5.MATRIX:raise ValueError('Expected nested matrix')
    if element.size==0:return numeric(np.empty((0,0)))
    flags,shape,_=mat5._header(element.payload,endian);cls=flags&255;count=math.prod(shape)
    if flags&0x800:raise ValueError('Complex data not supported by this audit')
    if cls==2:
        fields=names(element.payload,endian);rows=[]
        for _ in range(count):
            row={}
            for field in fields:
                child=mat5._Element(element.payload,endian)
                row[field]=v5_node(child,endian,path+'.'+field);child.finish()
            rows.append(row)
        return {'kind':'struct','shape':list(shape),'fields':fields,'rows':rows}
    if cls==1:
        items=[]
        for _ in range(count):
            child=mat5._Element(element.payload,endian);items.append(v5_node(child,endian,path+'[]'));child.finish()
        return {'kind':'cell','shape':list(shape),'items':items}
    if cls==4:
        if not count:return {'kind':'char','shape':list(shape),'text':''}
        kind,data=mat5._small_field(element.payload,endian)
        codecs={1:'latin1',2:'latin1',4:'utf-16-le' if endian=='<' else 'utf-16-be',16:'utf-8',17:'utf-16-le' if endian=='<' else 'utf-16-be',18:'utf-32-le' if endian=='<' else 'utf-32-be'}
        return {'kind':'char','shape':list(shape),'text':data.decode(codecs[kind]).rstrip('\0')}
    if cls not in range(6,16):raise ValueError(f'Unsupported MATLAB class {cls}')
    if not count:return numeric(np.empty(shape))
    data=mat5._Element(element.payload,endian)
    dtype=np.dtype(endian+mat5.DTYPES[data.kind])
    if data.size!=count*dtype.itemsize:raise ValueError('Numeric length mismatch')
    if path.endswith('.rawData') and count>1:
        data.finish();return {'kind':'numeric','shape':list(shape),'size':count,'dtype':str(dtype),'values_not_read':True}
    if count<=4096:
        a=np.frombuffer(data.read_bytes(32768),dtype=dtype).reshape(shape,order='F');out=numeric(a)
    else:
        out={'kind':'numeric','shape':list(shape),'size':count,'dtype':str(dtype),'finite':0,'nan':0,'inf':0}
        while data.payload.remaining:
            a=np.frombuffer(mat5._exact(data.payload,min(mat5.CHUNK,data.payload.remaining)),dtype=dtype)
            out['finite']+=int(np.isfinite(a).sum());out['nan']+=int(np.isnan(a).sum());out['inf']+=int(np.isinf(a).sum())
    data.finish();return out


def v5_sentences(path):
    with path.open('rb') as source:
        header=mat5._exact(source,128);endian={b'IM':'<',b'MI':'>'}[header[126:128]]
        outer=mat5._Element(source,endian)
        inflater=mat5._Inflater(outer.payload) if outer.kind==mat5.COMPRESSED else None
        matrix=mat5._Element(inflater,endian) if inflater else outer
        flags,shape,name=mat5._header(matrix.payload,endian)
        if name!='sentenceData' or flags&255!=2:raise ValueError('Expected top-level sentenceData struct')
        fields=names(matrix.payload,endian)
        for i in range(math.prod(shape)):
            row={}
            for field in fields:
                element=mat5._Element(matrix.payload,endian)
                row[field]=v5_node(element,endian,field if field.startswith('.') else '.'+field);element.finish()
            yield i,row
        if matrix.payload.remaining:raise ValueError('Trailing struct bytes')
        matrix.finish()
        if inflater:inflater.finish();outer.finish()


def hdf_node(f,node,path=''):
    if isinstance(node,h5py.Group):
        fields=list(node)
        # Word structs have reference columns, whereas allFixations is a scalar
        # struct containing direct numeric vectors. Never equate vector length
        # with the number of scalar structs.
        if fields and all(isinstance(node[k],h5py.Dataset) and h5py.check_dtype(ref=node[k].dtype) is not None for k in fields):
            refs={k:node[k][()].flatten(order='F') for k in fields}
            lengths={len(v) for v in refs.values()}
            if len(lengths)!=1:raise ValueError('Struct reference columns disagree')
            rows=[{k:hdf_node(f,f[refs[k][i]],path+'.'+k) for k in fields} for i in range(next(iter(lengths)))]
            return {'kind':'struct','fields':fields,'rows':rows,'shape':list(node[fields[0]].shape[::-1])}
        return {'kind':'struct','fields':fields,'rows':[{k:hdf_node(f,node[k],path+'.'+k) for k in fields}],'shape':[1,1]}
    if node.attrs.get('MATLAB_empty',0):
        return numeric(np.empty(tuple(int(x) for x in node[()])))
    if node.attrs.get('MATLAB_class')==b'char':
        return {'kind':'char','shape':list(node.shape[::-1]),'text':''.join(chr(int(v)) for v in node[()].flatten(order='F'))}
    if h5py.check_dtype(ref=node.dtype) is not None:
        return {'kind':'cell','shape':list(node.shape[::-1]),'items':[hdf_node(f,f[r],path+'[]') for r in node[()].flatten(order='F')]}
    count=int(node.size)
    if path.endswith('.rawData') and count>1:
        return {'kind':'numeric','shape':list(node.shape[::-1]),'dtype':str(node.dtype),'size':count,'values_not_read':True}
    if count>8*2**20//node.dtype.itemsize:
        out={'kind':'numeric','shape':list(node.shape[::-1]),'size':count,'dtype':str(node.dtype),'finite':0,'nan':0,'inf':0}
        block=max(1,mat5.CHUNK//node.dtype.itemsize//math.prod(node.shape[1:]))
        for start in range(0,node.shape[0],block):
            a=node[start:start+block];out['finite']+=int(np.isfinite(a).sum());out['nan']+=int(np.isnan(a).sum());out['inf']+=int(np.isinf(a).sum())
        return out
    return numeric(node[()].transpose(),node.shape[::-1])


def hdf_sentences(path):
    with h5py.File(path,'r') as f:
        s=f['sentenceData'];refs={k:s[k][()].flatten(order='F') for k in s}
        for i in range(len(refs['content'])):
            yield i,{k:hdf_node(f,f[values[i]],'.'+k) for k,values in refs.items()}


def status(node):
    if node is None:return 'absent'
    if node['kind']=='numeric':
        if node['size']==0:return 'empty'
        if node.get('nan')==node['size']:return 'nan_sentinel' if node['size']==1 else 'all_nan_vector'
        if node.get('nan',0) or node.get('inf',0):return 'partial_or_inf'
        return 'shape_only' if node.get('values_not_read') else 'finite'
    if node['kind']=='cell':return 'cell' if node['items'] else 'empty'
    return node['kind']


def scalar(node,default=None):
    if node and status(node)=='finite' and node.get('size')==1:return node['data'][0]
    return default


def vector(node,n):
    return node is not None and status(node)=='finite' and node['size']==n and sorted(node['shape'])==[1,n]


def field_stats(stats,name,node):
    s=stats.setdefault(name,{'states':Counter(),'shapes':Counter(),'dtypes':Counter()})
    s['states'][status(node)]+=1
    if node:
        s['shapes'][str(node['shape'])]+=1
        if 'dtype' in node:s['dtypes'][node['dtype']]+=1


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit_file(path,dataset,prepared,cohort):
    subject=path.stem[7:-3];start=time.monotonic();before=path.stat()
    total=Counter();sentence_stats={};word_stats={};raw_stats={};schemas=Counter();sentences=[];examples=[];mismatches=[];positions_bad=[]
    iterator=hdf_sentences(path) if h5py.is_hdf5(path) else v5_sentences(path)
    for index,s in iterator:
        text=s['content'].get('text','');w=s.get('word',{});words=w.get('rows',[])
        tokens=[x.get('content',{}).get('text','') for x in words]
        row={'dataset':dataset,'subject':subject,'sentence_id':str(index),'words':len(words),
             'prepared':(subject,str(index)) in prepared,'qc_retained':(subject,str(index)) in cohort,
             'raw_sentence_state':status(s.get('rawData')),'joined_word_text_exact':' '.join(tokens)==text,
             'literal_space_tokens_exact':text.split(' ')==tokens,'whitespace_tokens_exact':text.split()==tokens,
             'wordbounds_shape':s.get('wordbounds',{}).get('shape'), 'wordbounds_match':s.get('wordbounds',{}).get('shape',[0])[0]==len(words),
             'raw_fixations':0,'usable_raw_fixations':0,'words_with_usable_raw':0,
             'fixated_words':0,'omission_rate':scalar(s.get('omissionRate')),
             **{m+'_complete_8x105_words':0 for m in MEASURES}}
        if (subject,str(index)) in prepared and prepared[subject,str(index)]!=text:raise ValueError('Prepared target differs from source')
        allpos=[]
        for name,node in s.items():field_stats(sentence_stats,name,node)
        allfix=s.get('allFixations',{}).get('rows',[{}])
        row['allFixations_fields']=sorted(allfix[0]) if allfix else []
        for wi,word in enumerate(words):
            total['words']+=1;schemas[json.dumps(sorted(word))]+=1
            for name,node in word.items():field_stats(word_stats,name,node)
            for m in MEASURES:
                row[m+'_complete_8x105_words']+=all(vector(word.get(m+'_'+b),105) for b in BANDS)
            nfix=scalar(word.get('nFixations'),0)
            if nfix>0:row['fixated_words']+=1
            pos=word.get('fixPositions',{}).get('data',[]);allpos+=pos
            raw=word.get('rawEEG',{});cells=raw.get('items',[])
            row['raw_fixations']+=len(cells);usable=0
            for cell in cells:
                field_stats(raw_stats,'rawEEG_fixation',cell)
                good=status(cell)=='finite' and len(cell['shape'])==2 and cell['shape'][0]==105 and cell['shape'][1]>=2
                usable+=good
            if nfix!=len(cells) or nfix!=len(pos):
                if len(positions_bad)<100:positions_bad.append({'sentence':index,'word':wi,'nFixations':nfix,'raw_cells':len(cells),'fixPositions':pos})
                total['fixation_count_mismatch_words']+=1
            row['usable_raw_fixations']+=usable;row['words_with_usable_raw']+=usable>0
            if len(examples)<4 and (usable or nfix==0):
                examples.append({'sentence_id':index,'sentence_text':text,'word_index_one_based':wi+1,'word_content':tokens[wi],
                    'word_fields':sorted(word),'nFixations':nfix,'fixPositions':pos,'rawEEG_shapes':[c['shape'] for c in cells],
                    'rawEEG_states':[status(c) for c in cells],'rawET_shapes':[c['shape'] for c in word.get('rawET',{}).get('items',[])],
                    'FFD':scalar(word.get('FFD')),'TRT':scalar(word.get('TRT')),
                    'TRT_t1':{k:v for k,v in word.get('TRT_t1',{}).items() if k!='data'},
                    'allFixations_fields':row['allFixations_fields']})
        if allpos and sorted(allpos)!=list(range(1,len(allpos)+1)):total['noncontiguous_or_duplicate_word_fixation_positions_sentences']+=1
        if row['omission_rate'] is not None and words and not np.isclose(row['omission_rate'],1-row['fixated_words']/len(words)):
            total['omission_rate_count_mismatch_sentences']+=1
        if not row['joined_word_text_exact'] or not row['wordbounds_match']:
            mismatches.append({'sentence_id':index,'sentence_text':text,'words':tokens,'wordbounds_shape':row['wordbounds_shape']})
        sentences.append(row);total['sentences']+=1
        if row['joined_word_text_exact']:total['joined_word_text_exact']+=1
        if row['literal_space_tokens_exact']:total['literal_space_tokens_exact']+=1
        if row['whitespace_tokens_exact']:total['whitespace_tokens_exact']+=1
        for k in ['fixated_words','raw_fixations','usable_raw_fixations','words_with_usable_raw']+[m+'_complete_8x105_words' for m in MEASURES]:total[k]+=row[k]
    digest=sha(path)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during read-only audit')
    plan=json.loads((path.parent/'download_plan.json').read_text())
    expected=next(x['sha256'] for x in plan['files'] if x['name']==path.name)
    if digest!=expected:raise ValueError('Source checksum differs from pinned download plan')
    return {'dataset':dataset,'subject':subject,'source':str(path.resolve()),'sha256':digest,'source_matches_download_plan':True,
            'seconds':time.monotonic()-start,'counts':dict(total),'sentence_fields':sentence_stats,'word_fields':word_stats,
            'raw_fixation_fields':raw_stats,'word_struct_schemas':dict(schemas),'sentences':sentences,
            'mapping_mismatches':mismatches,'fixation_count_mismatch_examples':positions_bad,'examples':examples,
            'scope':'Full fields/shape/missingness/finite-value census of word EEG; rawData sentence matrix values skipped; no timing reconstructed'}


def main(dataset,output):
    if output.exists():raise FileExistsError('Use fresh audit output')
    root=Path(__file__).resolve().parents[1]
    output.mkdir(parents=True)
    prepared_path=root/f'data/processed/{dataset}_batch/combined.jsonl'
    rows=[json.loads(x) for x in prepared_path.read_text(encoding='utf-8').splitlines()]
    prepared={(r['subject'],r['sentence_id']):r['text'] for r in rows}
    qc=root/f'data/manifests/{dataset}_duration_qc.jsonl'
    cohort={(r['subject'],r['sentence_id']) for r in map(json.loads,qc.read_text(encoding='utf-8').splitlines())} if qc.exists() else set(prepared)
    summaries=[]
    for path in sorted((root/f'data/raw/{dataset}').glob('results*_NR.mat')):
        print('AUDIT '+path.name,flush=True)
        result=audit_file(path,dataset,prepared,cohort)
        (output/(result['subject']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        brief={k:v for k,v in result.items() if k in ('dataset','subject','source','sha256','seconds','counts')}
        summaries.append(brief)
        (output/'progress.json').write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(brief),flush=True)
    print('COMPLETE '+dataset,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',choices=['zuco1','zuco2'],required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.dataset,args.output)
