"""Synthetic MAT5 format/memory fixtures; compare selected values against SciPy."""
import gc
import struct
import tracemalloc
import zlib

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from eeg_decoding.data.mat5 import iter_sentence_fields
from eeg_decoding.data.zuco1 import iter_mat


def fixture(path, compressed=True, ignored_bytes=0, cell=False):
    fields = ['rawData','word','content','mean_t1']
    shape = (1,3)
    if cell:
        rows = np.empty(shape,dtype=object)
    else:
        rows = np.empty(shape,dtype=[(f,'O') for f in fields])
    for i in range(3):
        raw = np.nan if i == 1 else np.arange(3*(80+i),dtype=np.float32).reshape(3,80+i)/10
        row = {'rawData':raw,'content':f'Synthetic café — sentence {i}',
               'word':{'nested':np.zeros(max(1,ignored_bytes//8),dtype=np.float64)},
               'mean_t1':np.arange(3,dtype=float)}
        if cell:
            rows[0,i]=row
        else:
            for key,value in row.items():
                rows[key][0,i]=value
    savemat(path, {'unused_before':np.arange(9),'sentenceData':rows,'unused_after':'other variable'},
            do_compression=compressed,long_field_names=True)


@pytest.mark.parametrize('compressed',[False,True])
@pytest.mark.parametrize('cell',[False,True])
def test_streaming_matches_scipy_selected_fields(tmp_path,compressed,cell):
    path=tmp_path/'synthetic.mat'
    fixture(path,compressed,cell=cell)
    expected=loadmat(path,simplify_cells=True)['sentenceData']
    audit={}
    rows=list(iter_sentence_fields(path,audit=audit))
    assert audit['total_sentences']==3 and audit['reader']=='mat5_incremental'
    for (index,signal,text),source in zip(rows,expected):
        np.testing.assert_equal(signal.squeeze(),np.asarray(source['rawData']).squeeze())
        assert text==source['content'] and index in range(3)
    schema=dict(subject='fixture',session='unknown',task='nr',channels=['A','B','C'],sfreq=500,
                signal_field='rawData',text_field='content',time_axis=1,units='uV')
    imported=list(iter_mat(path,schema,tmp_path/'cache',skip_missing_eeg=True))
    assert [r.sentence_id for r in imported]==['0','2']


def test_large_ignored_payload_has_bounded_python_memory(tmp_path):
    path=tmp_path/'compressed_nested_fields.mat'
    fixture(path,True,ignored_bytes=16*2**20)
    gc.collect()
    tracemalloc.start()
    try:
        rows=list(iter_sentence_fields(path))
        _,peak=tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(rows)==3
    # 48 MiB of nested word data is traversed, not materialized. This bound is
    # independent of the total nested payload and allows interpreter overhead.
    assert peak < 8*2**20, f'Unexpected materialization: {peak/2**20:.1f} MiB'


def tag(kind,data,endian='<',small=False):
    if small and 0 < len(data) <= 4:
        return struct.pack(endian+'I',(len(data)<<16)|kind)+data.ljust(4,b'\0')
    return struct.pack(endian+'II',kind,len(data))+data+b'\0'*((-len(data))%8)


def matrix_header(cls,shape,name,endian):
    return (tag(6,struct.pack(endian+'II',cls,0),endian)+
            tag(5,struct.pack(endian+'i'*len(shape),*shape),endian)+
            tag(1,name.encode(),endian,small=True))


def file_header(endian):
    return (b'MATLAB 5.0 MAT-file, synthetic format fixture'.ljust(116,b' ')+b'\0'*8+
            struct.pack(endian+'H',0x100)+(b'IM' if endian=='<' else b'MI'))


def handcrafted(endian):
    values=np.arange(24,dtype=np.float32).reshape(3,8)
    numeric=matrix_header(7,(3,8),'',endian)+tag(7,values.astype(endian+'f4').tobytes(order='F'),endian)
    text='EEG café'
    chars=matrix_header(4,(1,len(text)),'',endian)+tag(17,text.encode('utf-16-le' if endian=='<' else 'utf-16-be'),endian)
    names=b'rawData\0content\0'
    body=(matrix_header(2,(1,1),'sentenceData',endian)+tag(5,struct.pack(endian+'i',8),endian,small=True)+
          tag(1,names,endian)+tag(14,numeric,endian)+tag(14,chars,endian))
    return tag(14,body,endian),values,text


@pytest.mark.parametrize('endian',['<','>'])
@pytest.mark.parametrize('compressed',[False,True])
def test_endian_small_tags_and_char_encoding_against_scipy(tmp_path,endian,compressed):
    body,values,text=handcrafted(endian)
    if compressed:
        data=zlib.compress(body)
        body=struct.pack(endian+'II',15,len(data))+data
    path=tmp_path/'endian.mat'
    path.write_bytes(file_header(endian)+body)
    source=loadmat(path,simplify_cells=True)['sentenceData']
    np.testing.assert_array_equal(source['rawData'],values)
    assert source['content']==text
    actual=list(iter_sentence_fields(path))
    np.testing.assert_array_equal(actual[0][1],values)
    assert actual[0][2]==text


def test_selected_field_limit_stays_enforced(tmp_path):
    path=tmp_path/'limited.mat'
    fixture(path,True)
    with pytest.raises(ValueError,match='materialization limit'):
        list(iter_sentence_fields(path,max_field_bytes=128))


def test_truncation_and_zlib_checksum_failure_are_rejected(tmp_path):
    body,_,_=handcrafted('<')
    compressed=zlib.compress(body)
    good=file_header('<')+struct.pack('<II',15,len(compressed))+compressed
    path=tmp_path/'bad.mat'
    for data in [good[:-3], good[:-1]+bytes([good[-1]^1])]:
        path.write_bytes(data)
        with pytest.raises(ValueError,match='Truncated|integrity'):
            list(iter_sentence_fields(path))
    payload=compressed+b'x'
    path.write_bytes(file_header('<')+struct.pack('<II',15,len(payload))+payload)
    with pytest.raises(ValueError,match='trailing bytes'):
        list(iter_sentence_fields(path))


def test_file_above_old_limit_without_full_materialization(tmp_path):
    body,values,text=handcrafted('<')
    path=tmp_path/'large_sparse.mat'
    # A valid second numeric variable makes this a >512 MiB Level 5 file.
    # Sparse filesystem extent avoids allocating its unused 513 MiB payload.
    size=513*2**20
    meta=matrix_header(9,(1,size),'unused_large_variable','<')
    with path.open('wb') as f:
        f.write(file_header('<')+body)
        f.write(struct.pack('<II',14,len(meta)+8+size))
        f.write(meta)
        f.write(struct.pack('<II',2,size))
        f.seek(size-1,1)
        f.write(b'\0')
    assert path.stat().st_size>512*2**20
    actual=list(iter_sentence_fields(path))
    np.testing.assert_array_equal(actual[0][1],values)
    assert actual[0][2]==text


def test_non_struct_text_and_missing_fields_fail_explicitly(tmp_path):
    path=tmp_path/'wrong.mat'
    savemat(path,{'sentenceData':np.arange(10)},do_compression=True)
    with pytest.raises(ValueError,match='struct'):
        list(iter_sentence_fields(path))
    savemat(path,{'sentenceData':{'content':'text','wrong_signal':np.ones((3,4))}},do_compression=True)
    with pytest.raises(ValueError,match='Missing/duplicate'):
        list(iter_sentence_fields(path))


def test_scalar_struct_custom_fields_and_writable_volts(tmp_path):
    path=tmp_path/'scalar.mat'
    signal_field='waveform_with_a_long_explicit_schema_field'
    text_field='sentence_source_text'
    volts=np.full((3,20),2e-6,np.float32)
    savemat(path,{'sentenceData':{signal_field:volts,text_field:'One synthetic sentence'}},
            do_compression=True,long_field_names=True)
    schema=dict(subject='fixture',session='unknown',task='nr',channels=['A','B','C'],sfreq=500,
                signal_field=signal_field,text_field=text_field,time_axis=1,units='V')
    records=list(iter_mat(path,schema,tmp_path/'cache'))
    assert len(records)==1 and records[0].sentence_id=='0'
    with np.load(records[0].signal_path) as f:
        np.testing.assert_allclose(f['eeg'],2.,rtol=1e-6)


def test_struct_indices_follow_matlab_column_major_order(tmp_path):
    path=tmp_path/'matrix_struct.mat'
    rows=np.empty((2,2),dtype=[('content','O'),('rawData','O')])
    for index in range(4):
        slot=np.unravel_index(index,(2,2),order='F')
        rows['content'][slot]=f'Sentence {index}'
        rows['rawData'][slot]=np.full((3,20),index,np.float32)
    savemat(path,{'sentenceData':rows},do_compression=True)
    result=list(iter_sentence_fields(path))
    assert [i for i,_,_ in result]==[0,1,2,3]
    assert [text for _,_,text in result]==[f'Sentence {i}' for i in range(4)]
    for index,signal,_ in result:
        np.testing.assert_array_equal(signal,np.full((3,20),index))
