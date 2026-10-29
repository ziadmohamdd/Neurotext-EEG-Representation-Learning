"""Frozen feature extraction into memory-mapped files; pretrained weights stay frozen."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .data.dataset import read_manifest, load_signal, collate_variable, observation_seed
from .data.validation import manifest_digest
from .models.pretrained_eeg import FrozenLaBraM, acquire_official, COMMIT, WEIGHTS_SHA256, file_sha256
from .preprocessing.quality import channel_reliability
from .evaluation.controls import gaussian_input, shuffled_keys
from .evaluation.robustness import corrupt
from .utils.config import load_config
from .utils.device import device_info
from .utils.logging import write_json

def observation_id(r):
    return hashlib.sha256(json.dumps([r.dataset,r.subject,r.session,r.task,r.sentence_id]).encode()).hexdigest()

def variant_name(control="real", corruption=None):
    return f"{corruption[0]}_{corruption[1]:g}" if corruption else ("gaussian" if control == "gaussian" else "real")

class FeatureCache:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path/"metadata.json").read_text())
        if self.metadata.get("status") != "complete":
            raise ValueError("Incomplete feature extraction cache")
        self.index = {key:i for i,key in enumerate(self.metadata["observation_ids"])}
        self.arrays = {}

    def matrix(self, records, variant="real"):
        if variant not in self.metadata["variants"]:
            raise ValueError(f"Feature cache missing {variant}; re-extract with --include-controls")
        if variant not in self.arrays:
            self.arrays[variant] = np.load(self.path/(variant+".npy"),mmap_mode="r",allow_pickle=False)
        return self.arrays[variant][[self.index[observation_id(r)] for r in records]]

    def verify(self, records, quality_aware=None, seed=None):
        if manifest_digest(records) != self.metadata["manifest_digest"]:
            raise ValueError("Feature cache belongs to different data or preprocessing")
        if quality_aware is not None and quality_aware != self.metadata["quality_aware"]:
            raise ValueError("Quality ablation and pretrained feature cache disagree")
        if seed is not None and self.metadata.get("seed") != seed:
            raise ValueError("Pretrained corruption seed differs: re-extract features for this experiment seed")

def verify_real_reuse(source, expected):
    """Require identical scientific inputs before copying already extracted features."""
    previous = FeatureCache(source)
    for key in ['model','commit','weights_sha256','manifest_digest','observation_ids',
                'quality_aware','mapping','seed','units','max_samples']:
        if key not in previous.metadata or previous.metadata[key] != expected[key]:
            raise ValueError(f'Real feature reuse mismatch: {key}')
    if 'real' not in previous.metadata['variants']:
        raise ValueError('Reuse source has no real features')
    path = previous.path/'real.npy'
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    if matrix.shape != (len(expected['observation_ids']),200) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError('Reuse source real features have invalid shape, dtype or values')
    return path, file_sha256(path)


def extract_features(args):
    output = Path(args.output)
    if (output/'metadata.json').exists():
        raise FileExistsError('Feature output already has metadata; choose a new output directory')
    records, config = read_manifest(args.manifest), load_config(args.config)
    if getattr(args,"seed",None) is not None:
        config["seed"] = args.seed
    if any(r.sfreq != 200 or r.channels != records[0].channels for r in records):
        raise ValueError("LaBraM cache requires harmonized channels at 200 Hz")
    if any(r.provenance.get("units") != "uV" for r in records):
        raise ValueError("LaBraM requires documented microvolt input, without train normalization")
    mapping = json.loads(Path(args.channel_mapping).read_text())
    device,info = device_info(config.get("device","auto"))
    cache = acquire_official(args.cache)
    model = FrozenLaBraM(cache,records[0].channels,mapping,200).to(device)
    output.mkdir(parents=True,exist_ok=True)
    variants = [("real",None),("gaussian",None)] if args.include_controls or getattr(args,'include_gaussian',False) else [("real",None)]
    if args.include_controls:
        for kind,levels in [("additive_noise",[0,.25,.5,1]),("channel_dropout",[0,.1,.25,.5]),("temporal_mask",[0,.1,.25,.5])]:
            variants += [("real",(kind,severity)) for severity in levels]
    meta = {"status":"extracting", "model":"official_LaBraM_base", "commit":COMMIT,"weights_sha256":WEIGHTS_SHA256,
            "manifest_digest":manifest_digest(records),"observation_ids":[observation_id(r) for r in records],
            "quality_aware":config.get("quality_aware",False),"device":info,"mapping":mapping,
            "seed":config.get("seed",42),
            "units":"uV_divided_by_100", "max_samples":config.get("max_samples",1600), "variants":[]}
    reused = None
    if getattr(args,'reuse_real_features',None):
        reused, checksum = verify_real_reuse(args.reuse_real_features, meta)
        meta['real_feature_reuse'] = {'source':str(reused.resolve()), 'sha256':checksum}
    write_json(output/"metadata.json",meta)
    torch.manual_seed(config.get("seed",42))
    with torch.inference_mode():
        for control,corruption in variants:
            name = variant_name(control,corruption)
            if name=='real' and reused is not None:
                shutil.copyfile(reused, output/'real.npy')
                if file_sha256(output/'real.npy') != meta['real_feature_reuse']['sha256']:
                    raise ValueError('Reused feature copy failed integrity verification')
                meta['variants'].append('real')
                print(f'Reused verified real LaBraM features: {len(records)} observations', flush=True)
                continue
            print(f'Extracting LaBraM {name}: {len(records)} observations on {device}', flush=True)
            array = np.lib.format.open_memmap(output/(name+".npy"),mode="w+",dtype=np.float32,shape=(len(records),200))
            for i,r in enumerate(records):
                x = load_signal(r)
                maximum = config.get("max_samples",1600)
                if maximum and x.shape[-1] > maximum:
                    start = (x.shape[-1]-maximum)//2
                    x = x[:,start:start+maximum]
                rng = np.random.default_rng(observation_seed(r,config.get("seed",42)))
                if control == "gaussian":
                    x = gaussian_input(x,rng)
                if corruption:
                    x = corrupt(x,*corruption,rng)
                tensor = torch.from_numpy(x.copy())[None].to(device)
                if config.get("quality_aware"):
                    tensor = tensor * channel_reliability(tensor)
                array[i] = model(tensor).cpu().numpy()[0]
                if (i+1) % 50 == 0 or i+1 == len(records):
                    print(f'{name}: {i+1}/{len(records)} observations', flush=True)
            array.flush()
            meta["variants"].append(name)
    meta["status"] = "complete"
    write_json(output/"metadata.json",meta)
    print(f"Extracted {len(records)} observations with verified frozen LaBraM; {len(variants)} conditions")

class FeatureDataset(Dataset):
    def __init__(self, records, features, config, control="real", scaler=None):
        self.records = records
        self.x = features.matrix(records,variant_name(control))
        if scaler:
            self.x = ((self.x-np.array(scaler["mean"])) / np.array(scaler["std"])).astype(np.float32)
        self.keys = shuffled_keys([r.key for r in records],config.get("seed",42)) if control == "shuffled" else [r.key for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        return {"eeg":torch.from_numpy(self.x[index].copy())[:,None],"key":self.keys[index],"index":index}

class FeatureIdentity(nn.Module):
    def forward(self,x,mask=None):
        return x.squeeze(-1)

def feature_loader(records,features,config,control,training,scaler):
    from .workflow import DistinctSentenceBatches
    dataset = FeatureDataset(records,features,config,control,scaler)
    batches = DistinctSentenceBatches(dataset.keys,config.get("batch_size",2),training,config.get("seed",42))
    return DataLoader(dataset,batch_sampler=batches,num_workers=0,collate_fn=collate_variable)
