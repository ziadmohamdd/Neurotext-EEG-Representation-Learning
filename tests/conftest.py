"""SYNTHETIC fixtures only. These are not ZuCo and cannot produce research results."""
import numpy as np
import pytest
import torch
from eeg_decoding.data.dataset import Record

@pytest.fixture(autouse=True)
def small_cpu():
    torch.set_num_threads(2)

@pytest.fixture
def records(tmp_path):
    rows = []
    rng = np.random.default_rng(7)
    for subject in range(6):
        for sentence in range(10):
            path = tmp_path/f"s{subject}_t{sentence}.npz"
            x = rng.normal(size=(3,220+sentence*7)).astype(np.float32)
            np.savez_compressed(path,eeg=x)
            rows.append(Record("zuco1",str(subject),"1","nr",str(sentence),f"Synthetic test sentence {sentence}",str(path),
                               ["C3","CZ","C4"],200.,provenance={"units":"uV","synthetic":True,"person_id":f"person{subject}"}))
    return rows

@pytest.fixture
def config():
    return {"seed":42,"device":"cpu","batch_size":2,"gradient_accumulation_steps":3,
            "embedding_dim":128,"num_workers":0,"pin_memory":False,"mixed_precision":"fp16_if_supported",
            "max_samples":240,"epochs":2,"patience":3,"learning_rate":.001,"gradient_clip":1.,"quality_aware":False}

@pytest.fixture
def text_cache(records,tmp_path):
    import json
    from eeg_decoding.training.alignment_train import TextCache
    keys = sorted({r.key for r in records})
    # Random text vectors solely test tensor plumbing; never a substitute for language embeddings.
    vectors = np.random.default_rng(3).normal(size=(len(keys),12)).astype(np.float32)
    path = tmp_path/"SYNTHETIC_text.npz"
    np.savez_compressed(path,keys=np.array(keys),embeddings=vectors,metadata=json.dumps({"synthetic":True}))
    return TextCache(path)
