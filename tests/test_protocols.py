from dataclasses import replace
import pytest
from eeg_decoding.evaluation.cross_subject import group_split,check_disjoint,loso
from eeg_decoding.evaluation.cross_dataset import compatibility,transfer_split
from eeg_decoding.evaluation.controls import shuffled_keys

def test_subject_splits(records):
    splits=group_split(records)
    check_disjoint(splits)
    assert sum(map(len,splits.values()))==len(records)
    bad=dict(splits,val=splits["train"])
    with pytest.raises(ValueError,match="Subject leakage"):
        check_disjoint(bad)

def test_sentence_leakage(records):
    splits=group_split(records,held_out_sentences=True)
    check_disjoint(splits,sentences=True)
    with pytest.raises(ValueError,match="Sentence leakage"):
        check_disjoint(group_split(records),sentences=True)

def test_loso(records):
    folds=list(loso(records))
    assert len(folds)==6
    for subject,split in folds:
        assert {r.subject_key for r in split["test"]}=={subject}

def test_transfer_compatibility(records):
    source=[r for r in records if int(r.subject)<4]
    target=[replace(r,dataset="zuco2") for r in records if int(r.subject)>=4]
    report=compatibility(source,target)
    assert report["sentence_overlap"]==10
    assert transfer_split(source,target)["test"]==target
    for bad in [[replace(r,sfreq=500) for r in target], [replace(r,task="tsr") for r in target],
                [replace(r,provenance={}) for r in target], [replace(r,provenance={"person_id":"person0"}) for r in target]]:
        with pytest.raises(ValueError):
            compatibility(source,bad)

def test_shuffled_pairs_duplicates():
    keys=["a","a","b","b","c"]
    shuffled=shuffled_keys(keys)
    assert all(a!=b for a,b in zip(keys,shuffled))
    assert shuffled[0]==shuffled[1]
