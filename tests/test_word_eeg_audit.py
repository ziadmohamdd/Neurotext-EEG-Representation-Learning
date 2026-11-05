import importlib.util
from pathlib import Path
import h5py
import numpy as np
from scipy.io import savemat

spec=importlib.util.spec_from_file_location('word_audit',Path(__file__).parents[1]/'scripts/audit_word_eeg.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_nested_level5_struct_cells_and_missing_values(tmp_path):
    cells=np.empty((1,2),dtype=object);cells[0,0]=np.ones((105,31));cells[0,1]=np.array([[np.nan]])
    p=tmp_path/'nested.mat'
    savemat(p,{'sentenceData':{'content':'A word.','rawData':np.ones((105,200)),
        'word':np.array([{'content':'A','rawEEG':cells,'TRT_t1':np.arange(105.)[None]},
                         {'content':'word.','rawEEG':np.empty((0,0)),'TRT_t1':np.full((1,105),np.nan)}],dtype=object)}},do_compression=True)
    _,s=next(audit.v5_sentences(p))
    assert s['rawData']['values_not_read']
    words=s['word']['items']
    first=words[0]['rows'][0];last=words[1]['rows'][0]
    assert first['content']['text']=='A'
    assert first['rawEEG']['items'][0]['shape']==[105,31]
    assert audit.status(first['rawEEG']['items'][1])=='nan_sentinel'
    assert audit.vector(first['TRT_t1'],105)
    assert audit.status(last['TRT_t1'])=='all_nan_vector'


def test_hdf_matlab_axes_and_empty_dimensions(tmp_path):
    with h5py.File(tmp_path/'axes.h5','w') as f:
        d=f.create_dataset('x',data=np.arange(210).reshape(2,105))
        n=audit.hdf_node(f,d)
        assert n['shape']==[105,2]
        assert n['data']==list(range(210))
        e=f.create_dataset('empty',data=np.array([0,0],dtype='u8'));e.attrs['MATLAB_empty']=1
        assert audit.status(audit.hdf_node(f,e))=='empty'


def test_hdf_direct_vector_fields_remain_one_scalar_struct(tmp_path):
    with h5py.File(tmp_path/'struct.h5','w') as f:
        g=f.create_group('allFixations');g.create_dataset('duration',data=np.arange(4)[:,None]);g.create_dataset('x',data=np.arange(4)[:,None])
        n=audit.hdf_node(f,g)
        assert len(n['rows'])==1 and n['rows'][0]['duration']['shape']==[1,4]


def test_finite_vector_requires_exact_shape_and_all_finite():
    assert audit.vector(audit.numeric(np.ones((1,105))),105)
    assert not audit.vector(audit.numeric(np.ones((3,35))),105)
    assert not audit.vector(audit.numeric(np.full((1,105),np.nan)),105)
    assert audit.status(audit.numeric(np.array([1.,np.nan])))=='partial_or_inf'


def test_exact_containment_does_not_assign_an_unmatched_window(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    from probe_word_eeg_containment import exact_offsets
    sentence=np.arange(30).reshape(3,10)
    assert exact_offsets(sentence,sentence[:,3:7])==[3]
    assert exact_offsets(sentence,np.full((3,4),np.nan))==[]
    assert exact_offsets(sentence,sentence[:,3:7]+.1)==[]
