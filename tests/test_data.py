from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.io import savemat
from eeg_decoding.data.dataset import *
from eeg_decoding.data.validation import validate_records
from eeg_decoding.data.zuco1 import iter_mat
from eeg_decoding.data.zuco2 import iter_mat as iter_mat2
from eeg_decoding.preprocessing.segmentation import sample_bounds, harmonize

def test_manifest_roundtrip(records,tmp_path):
    path = tmp_path/"manifest.jsonl"
    write_manifest(path,records)
    loaded = read_manifest(path)
    assert loaded[0] == records[0]
    assert validate_records(loaded)["valid"]

def test_sentence_identity():
    assert sentence_key("  Same SENTENCE ") == sentence_key("same sentence")
    with pytest.raises(ValueError):
        sentence_key("  ")

def test_alignment_bounds():
    assert sample_bounds(1,.5,200,500) == (200,300)
    for onset,duration in [(-1,1),(0,0),(2,3),(float('nan'),1)]:
        with pytest.raises(ValueError):
            sample_bounds(onset,duration,200,500)

def test_variable_batch(records):
    dataset = EEGDataset(records,max_samples=None)
    batch = collate_variable([dataset[0],dataset[9]])
    assert batch["eeg"].shape == (2,3,283)
    assert batch["mask"].sum(1).tolist() == [220,283]
    assert not batch["eeg"][0,:,220:].any()

def test_channel_mismatch(records):
    with pytest.raises(ValueError):
        EEGDataset([records[0],replace(records[1],channels=["C4","C3","CZ"])])
    with pytest.raises(ValueError):
        harmonize(np.zeros((2,10)),["A","B"],["C"])
    assert harmonize(np.array([[1,2],[3,4]]),["A","B"],["B","A"])[0,0] == 3

def test_invalid_signal(records,tmp_path):
    path = tmp_path/"nan.npz"
    np.savez_compressed(path,eeg=np.full((3,10),np.nan))
    assert not validate_records([replace(records[0],signal_path=str(path))])["valid"]
    assert not validate_records([records[0],records[0]])["valid"]

def test_mat_v5_parsing(tmp_path):
    path = tmp_path/"SYNTHETIC.mat"
    data = np.empty(2,dtype=object)
    data[0] = {"rawData":np.ones((3,80)),"content":"first synthetic sentence"}
    data[1] = {"rawData":np.ones((3,90))*2,"content":"second synthetic sentence"}
    savemat(path,{"sentenceData":data})
    metadata = dict(subject="test",session="1",task="nr",channels=["A","B","C"],sfreq=200,
                    signal_field="rawData",text_field="content",time_axis=1,units="uV")
    for loader,version in [(iter_mat,"zuco1"),(iter_mat2,"zuco2")]:
        rows = list(loader(path,metadata,tmp_path/version))
        assert len(rows) == 2 and rows[0].dataset == version
        assert load_signal(rows[1]).shape == (3,90)
        assert rows[0].text == "first synthetic sentence"
    with pytest.raises(ValueError):
        list(iter_mat(path,{},tmp_path/"bad"))

def test_mat_hdf5_references(tmp_path):
    import h5py
    path = tmp_path/"SYNTHETIC_v73.mat"
    with h5py.File(path,"w") as f:
        group=f.create_group("sentenceData")
        a=f.create_dataset("signal",data=np.ones((80,3)))
        b=f.create_dataset("text",data=np.array([ord(c) for c in "synthetic text"])[:,None])
        x=group.create_dataset("rawData",(1,1),dtype=h5py.ref_dtype)
        t=group.create_dataset("content",(1,1),dtype=h5py.ref_dtype)
        x[0,0],t[0,0]=a.ref,b.ref
    meta=dict(subject="test",session="1",task="nr",channels=["A","B","C"],sfreq=200,
              signal_field="rawData",text_field="content",time_axis=1,units="uV")
    rows=list(iter_mat(path,meta,tmp_path/"hdf"))
    assert rows[0].text == "synthetic text" and load_signal(rows[0]).shape == (3,80)

def test_bids_requires_sentence_mapping(tmp_path):
    from eeg_decoding.data.nemar import iter_bids
    path=tmp_path/"events.tsv"
    path.write_text("onset\tduration\ttrial_type\n0\t1\ttrigger_12\n")
    with pytest.raises(ValueError,match="Verified sentence"):
        list(iter_bids(tmp_path,path,"zuco1",tmp_path/"cache"))

def test_bids_brainvision_roundtrip(tmp_path):
    # Actual BrainVision parser, tiny synthetic binary; not a research recording.
    from eeg_decoding.data.nemar import iter_bids
    x=np.arange(600,dtype=np.float32).reshape(200,3)
    x.tofile(tmp_path/"tiny.eeg")
    (tmp_path/"tiny.vmrk").write_text("Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile=tiny.eeg\n[Marker Infos]\n")
    (tmp_path/"tiny.vhdr").write_text("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile=tiny.eeg\nMarkerFile=tiny.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=3\nSamplingInterval=5000\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=C3,,1,µV\nCh2=CZ,,1,µV\nCh3=C4,,1,µV\n",encoding="utf-8")
    alignment=tmp_path/"alignment.tsv"
    alignment.write_text("recording\tonset\tduration\tsentence_id\ttext\tsubject\tsession\ttask\ntiny.vhdr\t0.1\t0.5\t0\tSynthetic sentence\tS1\t1\tnr\n")
    rows=list(iter_bids(tmp_path,alignment,"zuco1",tmp_path/"cache"))
    assert rows[0].channels==["C3","CZ","C4"]
    np.testing.assert_allclose(load_signal(rows[0]),x[20:120].T,atol=1e-3)

@pytest.mark.parametrize('version', ['zuco1', 'zuco2'])
def test_mat_missing_slots_preserve_sentence_ids(tmp_path, version):
    """Synthetic missing slots reproduce the user-observed ZuCo export pattern."""
    path = tmp_path / 'missing.mat'
    entries = np.empty(4, dtype=object)
    for i, signal in enumerate([np.nan, np.ones((3,80)), np.array([]), np.ones((3,90))]):
        entries[i] = {'rawData':signal, 'content':f'Synthetic sentence {i}'}
    savemat(path, {'sentenceData':entries})
    schema = dict(subject='fixture', session='1', task='nr', channels=['A','B','C'], sfreq=200,
                  signal_field='rawData', text_field='content', time_axis=1, units='uV')
    loader = iter_mat if version == 'zuco1' else iter_mat2
    with pytest.raises(ValueError, match='missing EEG'):
        list(loader(path, schema, tmp_path/'strict'))
    audit = {}
    rows = list(loader(path, schema, tmp_path/'skip', skip_missing_eeg=True, audit=audit))
    assert [r.sentence_id for r in rows] == ['1','3']
    assert [r.text for r in rows] == ['Synthetic sentence 1','Synthetic sentence 3']
    assert [r.provenance['matlab_sentence_index'] for r in rows] == [2,4]
    assert audit['total_sentences'] == 4 and audit['imported'] == 2
    assert [r['sentence_id'] for r in audit['skipped']] == ['0','2']
    entries[0] = {'rawData':np.ones((2,80)), 'content':'Wrong montage'}
    savemat(path, {'sentenceData':entries})
    with pytest.raises(ValueError, match='expected raw C x T'):
        list(loader(path, schema, tmp_path/'bad', skip_missing_eeg=True))


def test_hdf_missing_slot_and_nonfinite_waveform(tmp_path):
    import h5py
    path = tmp_path/'missing_hdf.mat'
    with h5py.File(path,'w') as f:
        group = f.create_group('sentenceData')
        signals = group.create_dataset('rawData',(1,3),dtype=h5py.ref_dtype)
        texts = group.create_dataset('content',(1,3),dtype=h5py.ref_dtype)
        for i in range(3):
            signal = np.array([[np.nan]]) if i == 0 else np.ones((80,3))
            signals[0,i] = f.create_dataset(f'signal{i}',data=signal).ref
            texts[0,i] = f.create_dataset(f'text{i}',data=np.array([ord(c) for c in f'Fixture {i}'])[:,None]).ref
    schema = dict(subject='fixture',session='1',task='nr',channels=['A','B','C'],sfreq=200,
                  signal_field='rawData',text_field='content',time_axis=1,units='uV')
    audit = {}
    rows = list(iter_mat(path,schema,tmp_path/'cache',skip_missing_eeg=True,audit=audit))
    assert [r.sentence_id for r in rows] == ['1','2']
    assert audit['imported'] == 2 and len(audit['skipped']) == 1
    with h5py.File(path,'a') as f:
        f['signal1'][0,0] = np.inf
    with pytest.raises(ValueError,match='NaN/Inf'):
        list(iter_mat(path,schema,tmp_path/'bad',skip_missing_eeg=True))
