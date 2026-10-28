"""Synthetic batch/acquisition fixtures only; these are not ZuCo research results."""
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from eeg_decoding.batch import prepare_batch
from eeg_decoding.cli import main
from eeg_decoding.data.acquisition import download_file, plan_download
from eeg_decoding.data.dataset import read_manifest, load_signal
from eeg_decoding.data.metadata import read_eeg_metadata


def header(path, labels=('E2','E3','Cz'), sfreq=500, hdf=False):
    if not hdf:
        locations = np.empty(len(labels), dtype=object)
        for i, label in enumerate(labels):
            locations[i] = {'labels':label}
        savemat(path, {'EEG':{'srate':sfreq, 'nbchan':len(labels), 'chanlocs':locations, 'history':'synthetic fixture'}})
    else:
        with h5py.File(path, 'w') as f:
            eeg = f.create_group('EEG')
            eeg.create_dataset('srate',data=[[sfreq]])
            eeg.create_dataset('nbchan',data=[[len(labels)]])
            locations = f.create_group('locations')
            refs = locations.create_dataset('labels',(len(labels),1),dtype=h5py.ref_dtype)
            for i, label in enumerate(labels):
                refs[i,0] = f.create_dataset(f'label{i}',data=np.array([ord(c) for c in label])[:,None]).ref
            eeg.create_dataset('chanlocs',(1,1),dtype=h5py.ref_dtype)[0,0] = locations.ref
            eeg.create_dataset('history',data=np.array([ord(c) for c in 'synthetic fixture'])[:,None])


def subject(root, name, labels=('E2','E3','Cz'), hdf_header=False):
    root.mkdir(exist_ok=True)
    rows = np.empty(4,dtype=object)
    for i in range(4):
        raw = np.nan if i == 0 else np.ones((len(labels),1000+i*10),np.float32)*i
        rows[i] = {'content':f'Synthetic sentence {i}', 'rawData':raw}
    savemat(root/f'results{name}_NR.mat',{'sentenceData':rows})
    header(root/f'oip_{name}_NR6_EEG.mat',labels,hdf=hdf_header)


@pytest.mark.parametrize('hdf',[False,True])
def test_read_ordered_eeg_metadata(tmp_path,hdf):
    path = tmp_path/'header.mat'
    header(path,hdf=hdf)
    meta = read_eeg_metadata(path)
    assert meta['channels'] == ['E2','E3','Cz'] and meta['sfreq'] == 500
    assert len(meta['sha256']) == 64 and meta['history'] == 'synthetic fixture'


@pytest.mark.parametrize('format', ['v5', 'hdf_refs', 'hdf_direct'])
def test_export_actual_electrode_coordinates(tmp_path, format):
    path = tmp_path/'oip_SYNTHETIC_NR1_EEG.mat'
    labels = ('E2','E3','Cz')
    expected = {'X':[1., 2., 3.], 'Y':[-1., -2., -3.], 'Z':[4., None, 6.]}
    if format == 'v5':
        locs = np.empty(3, dtype=object)
        for i, label in enumerate(labels):
            locs[i] = dict(labels=label, **{axis:values[i] if values[i] is not None else []
                                          for axis, values in expected.items()})
        savemat(path, {'EEG':{'srate':500, 'nbchan':3, 'chanlocs':locs}})
    else:
        header(path, labels, hdf=True)
        with h5py.File(path, 'a') as f:
            for axis, values in expected.items():
                if format == 'hdf_direct':
                    f['locations'].create_dataset(axis, data=np.array([
                        np.nan if v is None else v for v in values])[:,None])
                else:
                    refs = f['locations'].create_dataset(axis, (3,1), dtype=h5py.ref_dtype)
                    for i, value in enumerate(values):
                        data = [] if value is None else [[value]]
                        refs[i,0] = f.create_dataset(f'{axis}{i}', data=data).ref
    meta = read_eeg_metadata(path, include_coordinates=True)
    assert meta['electrodes'] == [dict(label=label, **{axis:values[i] for axis, values in expected.items()})
                                  for i, label in enumerate(labels)]
    json.dumps(meta, allow_nan=False)
    assert 'electrodes' not in read_eeg_metadata(path)


def test_coordinate_audit_cli_preserves_missing_positions(tmp_path, monkeypatch):
    from eeg_decoding.data.channel_audit import main as audit_main
    root = tmp_path/'raw'
    root.mkdir()
    header(root/'oip_SYNTHETIC_NR1_EEG.mat')
    output = tmp_path/'report.json'
    monkeypatch.setattr('sys.argv', ['channel_audit', '--roots', str(root), '--output', str(output)])
    audit_main()
    report = json.loads(output.read_text())
    assert report['status'] == 'coordinates_exported_not_mapping_verified'
    assert report['recordings'][0]['n_complete_xyz'] == 0
    assert report['recordings'][0]['electrodes'][0] == dict(label='E2', X=None, Y=None, Z=None)


def test_batch_all_subjects_repeatable_and_content_checked(tmp_path):
    root, output = tmp_path/'raw', tmp_path/'prepared'
    for i,name in enumerate(['ZAA','ZBB','ZCC']):
        subject(root,name,hdf_header=bool(i%2))
    report = prepare_batch(root,'zuco1',output,'uV',cache_text=False)
    assert report['status'] == 'prepared' and report['n_subjects'] == 3
    assert report['n_observations'] == 9 and report['n_unique_sentences'] == 3
    assert report['subject_split_possible']
    rows = read_manifest(report['manifest'])
    assert {r.sentence_id for r in rows} == {'1','2','3'}
    assert all(r.sfreq == 200 for r in rows)
    assert load_signal(rows[0]).shape == (3,404)
    assert not list(output.rglob('scratch/*.npz'))
    again = prepare_batch(root,'zuco1',output,'uV',cache_text=False)
    assert all(s['status']=='reused' for s in again['subjects'].values())
    with open(rows[0].signal_path,'ab') as f:
        f.write(b'changed')
    changed = prepare_batch(root,'zuco1',output,'uV',cache_text=False)
    assert changed['status']=='partial' and changed['failed_subjects']==['ZAA']
    assert Path(changed['manifest']).name=='partial.jsonl'


def test_batch_missing_metadata_and_montage_policy(tmp_path):
    root, output = tmp_path/'raw',tmp_path/'strict'
    subject(root,'ZAA')
    subject(root,'ZBB',labels=('E2','E3','E4'))
    subject(root,'ZCC')
    (root/'oip_ZCC_NR6_EEG.mat').unlink()
    report = prepare_batch(root,'zuco1',output,'uV',cache_text=False)
    assert report['status']=='partial' and sorted(report['failed_subjects'])==['ZBB','ZCC']
    report = prepare_batch(root,'zuco1',tmp_path/'intersection','uV',channel_policy='intersection',cache_text=False)
    assert report['channels']==['E2','E3'] and report['n_subjects']==2
    assert report['status']=='partial'


@pytest.mark.parametrize('hdf', [False, True])
def test_batch_audits_segments_too_short_for_resampling(tmp_path, hdf):
    root = tmp_path/'raw'
    root.mkdir()
    path = root/'resultsYTL_NR.mat'
    lengths = [1, 2, 3, 1000]
    if hdf:
        with h5py.File(path, 'w') as f:
            group = f.create_group('sentenceData')
            signals = group.create_dataset('rawData', (1,4), dtype=h5py.ref_dtype)
            texts = group.create_dataset('content', (1,4), dtype=h5py.ref_dtype)
            for i, n in enumerate(lengths):
                signals[0,i] = f.create_dataset(f'signal{i}', data=np.ones((n,3))).ref
                texts[0,i] = f.create_dataset(f'text{i}', data=np.array([ord(c) for c in f'Fixture {i}'])[:,None]).ref
    else:
        rows = np.empty(4, dtype=object)
        for i, n in enumerate(lengths):
            rows[i] = {'rawData':np.ones((3,n)), 'content':f'Fixture {i}'}
        savemat(path, {'sentenceData':rows})
    header(root/'oip_YTL_NR6_EEG.mat')
    report = prepare_batch(root, 'zuco2', tmp_path/'out', 'uV', cache_text=False)
    assert report['status'] == 'prepared'
    audit = report['subjects']['YTL']
    assert audit['imported'] == 2
    assert audit['skipped'] == [
        {'sentence_id':str(i), 'matlab_index':i+1,
         'reason':'insufficient_temporal_samples', 'n_samples':n, 'min_samples':3}
        for i, n in enumerate([1,2])
    ]
    records = read_manifest(report['manifest'])
    assert [r.sentence_id for r in records] == ['2','3']
    assert load_signal(records[0]).shape == (3,2)
    again = prepare_batch(root, 'zuco2', tmp_path/'out', 'uV', cache_text=False)
    assert again['subjects']['YTL']['status'] == 'reused'


def test_batch_verifies_all_supplied_runs_and_requested_subjects(tmp_path):
    root = tmp_path/'raw'
    subject(root,'ZAA')
    header(root/'oip_ZAA_NR5_EEG.mat',sfreq=250)
    report = prepare_batch(root,'zuco1',tmp_path/'out','uV',cache_text=False)
    assert report['status']=='failed' and 'disagree' in report['subjects']['ZAA']['error']
    with pytest.raises(ValueError,match='Missing requested'):
        prepare_batch(root,'zuco1',tmp_path/'missing','uV',subjects=['ZBB'],cache_text=False)


def test_cli_batch_one_command_and_shared_text_pool(tmp_path,monkeypatch):
    root = tmp_path/'raw'
    for name in ['ZAA','ZBB','ZCC']:
        subject(root,name)
    calls=[]
    def fake_text_build(records,path,model,revision):
        calls.append((len(records),len({r.key for r in records})))
        Path(path).write_bytes(b'SYNTHETIC_TEST_CACHE_ONLY')
    monkeypatch.setattr('eeg_decoding.batch.TextCache.build',fake_text_build)
    output=tmp_path/'out'
    main(['prepare-batch','--dataset','zuco1','--root',str(root),'--output',str(output),
          '--units','uV','--already-preprocessed'])
    report=json.loads((output/'summary.json').read_text())
    assert report['status']=='prepared' and calls==[(9,3)]
    assert Path(report['text_cache']).is_file()


class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data)
        self.status,self.headers=status,headers or {}


def entry(data):
    return {'name':'test.mat','size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'url':'https://osf.io/download/fixture/'}


def test_verified_download_reuse_and_preserved_user_file(tmp_path,monkeypatch):
    data=b'synthetic downloaded fixture'
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(data))
    assert download_file(entry(data),tmp_path)=='downloaded'
    assert download_file(entry(data),tmp_path)=='reused'
    (tmp_path/'test.mat').write_bytes(b'existing user file')
    with pytest.raises(FileExistsError,match='preserved'):
        download_file(entry(data),tmp_path)
    assert (tmp_path/'test.mat').read_bytes()==b'existing user file'


def test_download_resume_and_ignored_range(tmp_path,monkeypatch):
    data=b'1234567890'
    partial=tmp_path/'test.mat.partial'
    partial.write_bytes(data[:4])
    def resumed(request,**kw):
        assert request.get_header('Range')=='bytes=4-'
        return Response(data[4:],206,{'Content-Range':'bytes 4-9/10'})
    monkeypatch.setattr('urllib.request.urlopen',resumed)
    assert download_file(entry(data),tmp_path)=='downloaded'
    (tmp_path/'test.mat').unlink()
    partial.write_bytes(data[:4])
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(data))
    assert download_file(entry(data),tmp_path)=='downloaded'


def test_failed_verification_never_promotes_download(tmp_path,monkeypatch):
    data=b'correct bytes'
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(b'incorrect'))
    with pytest.raises(ValueError,match='checksum'):
        download_file(entry(data),tmp_path)
    assert not (tmp_path/'test.mat').exists()
    assert (tmp_path/'test.mat.partial').exists()


def test_official_plan_selects_smallest_metadata_and_checks_subjects(monkeypatch):
    def folder(name,path):
        return {'attributes':{'kind':'folder','name':name,'path':path}}
    def file(name,size):
        return {'attributes':{'kind':'file','name':name,'size':size,'extra':{'hashes':{'sha256':'a'*64}}},
                'links':{'download':'https://osf.io/download/fixture/'}}
    listings={'':[folder('task2 - NR','/nr/')],'/nr/':[folder('Matlab files','/exports/'),folder('Preprocessed','/prep/')],
        '/prep/':[folder('ZAA','/zaa/')],'/exports/':[file('resultsZAA_NR.mat',100)],
        '/zaa/':[file('oip_ZAA_NR1_EEG.mat',50),file('oip_ZAA_NR6_EEG.mat',10)]}
    monkeypatch.setattr('eeg_decoding.data.acquisition.osf_list',lambda node,folder='':listings[folder])
    plan=plan_download('zuco1')
    assert plan['total_bytes']==110 and plan['subjects']==['ZAA']
    assert plan['files'][1]['name']=='oip_ZAA_NR6_EEG.mat'
    with pytest.raises(ValueError,match='Unknown requested'):
        plan_download('zuco1',['NO_SUCH_SUBJECT'])


def test_download_and_prepare_cohort_in_one_cli_call(tmp_path,monkeypatch):
    authority=tmp_path/'authority'
    for name in ['ZAA','ZBB','ZCC']:
        subject(authority,name)
    payloads,files={},[]
    for path in sorted(authority.glob('*.mat')):
        data=path.read_bytes()
        url='https://osf.io/download/test_'+path.name
        payloads[url]=data
        files.append({'name':path.name,'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'url':url})
    plan={'dataset':'zuco1','subjects':['ZAA','ZBB','ZCC'],'files':files,'total_bytes':sum(len(d) for d in payloads.values())}
    monkeypatch.setattr('eeg_decoding.data.acquisition.plan_download',lambda *a,**kw:plan)
    monkeypatch.setattr('urllib.request.urlopen',lambda request,**kw:Response(payloads[request.full_url]))
    root,output=tmp_path/'downloaded',tmp_path/'prepared'
    command=['prepare-batch','--dataset','zuco1','--root',str(root),'--output',str(output),
             '--units','uV','--already-preprocessed','--download','--skip-text']
    main(command)
    report=json.loads((output/'summary.json').read_text())
    assert report['n_subjects']==3 and report['n_observations']==9 and report['status']=='prepared'
    assert json.loads((root/'download_report.json').read_text())['status']=='complete'
    main(command)
    assert all(s['status']=='reused' for s in json.loads((output/'summary.json').read_text())['subjects'].values())


def test_cli_reports_incomplete_cohort_with_nonzero_exit(tmp_path):
    root,output=tmp_path/'raw',tmp_path/'out'
    subject(root,'ZAA')
    subject(root,'ZBB')
    (root/'oip_ZBB_NR6_EEG.mat').unlink()
    with pytest.raises(SystemExit,match='incomplete'):
        main(['prepare-batch','--dataset','zuco1','--root',str(root),'--output',str(output),
              '--units','uV','--already-preprocessed','--skip-text'])
    report=json.loads((output/'summary.json').read_text())
    assert report['status']=='partial' and report['failed_subjects']==['ZBB']


def test_download_rejects_wrong_resume_range(tmp_path,monkeypatch):
    data=b'1234567890'
    partial=tmp_path/'test.mat.partial'
    partial.write_bytes(data[:4])
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Response(data[4:],206,{'Content-Range':'bytes 0-5/10'}))
    with pytest.raises(ValueError,match='partial response'):
        download_file(entry(data),tmp_path)
    assert partial.read_bytes()==data[:4]


def test_acquisition_checks_disk_before_downloading(tmp_path,monkeypatch):
    from types import SimpleNamespace
    plan={'subjects':['ZAA'],'files':[entry(b'fixture')],'total_bytes':7}
    monkeypatch.setattr('eeg_decoding.data.acquisition.plan_download',lambda *a,**kw:plan)
    monkeypatch.setattr('eeg_decoding.cli.shutil.disk_usage',lambda root:SimpleNamespace(free=0))
    with pytest.raises(SystemExit,match='Insufficient disk'):
        main(['download-zuco','--dataset','zuco1','--root',str(tmp_path),'--execute'])
