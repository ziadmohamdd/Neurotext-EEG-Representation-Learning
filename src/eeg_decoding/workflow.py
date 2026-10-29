"""Shared local/Colab orchestration. All model selection is validation-only."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from .data.dataset import EEGDataset, collate_variable, load_signal, observation_seed
from .data.validation import manifest_digest
from .evaluation.cross_subject import group_split, split_manifest
from .evaluation.cross_dataset import transfer_split, compatibility
from .evaluation.controls import gaussian_input, shuffled_keys
from .evaluation.robustness import corrupt
from .evaluation.retrieval import ranks, summarize, chance
from .evaluation.statistics import subject_bootstrap
from .models.eeg_cnn import EEGCNN
from .models.ssl_encoder import MaskedEEG
from .models.projection import EEGTextModel
from .models.classical import eeg_features, ridge_model
from .preprocessing.normalization import ChannelNormalizer
from .training.checkpoints import load_checkpoint
from .training.alignment_train import TextCache, train_alignment
from .training.ssl_train import train_ssl
from .utils.logging import write_json
from .utils.device import device_info, conservative_batch
from .utils.reproducibility import seed_all

class ObservationView(EEGDataset):
    def __init__(self, records, normalizer=None, max_samples=1600, control="real", corruption=None, seed=42):
        super().__init__(records, normalizer, max_samples)
        self.control, self.corruption, self.seed = control, corruption, seed
        self.training_keys = shuffled_keys([r.key for r in records], seed) if control == "shuffled" else None

    def __getitem__(self, index):
        r = self.records[index]
        x = load_signal(r)
        if self.max_samples and x.shape[1] > self.max_samples:
            start = (x.shape[1] - self.max_samples) // 2
            x = x[:, start:start+self.max_samples]
        rng = np.random.default_rng(observation_seed(r, self.seed))
        if self.control == "gaussian":
            x = gaussian_input(x, rng)
        if self.corruption:
            x = corrupt(x, *self.corruption, rng)
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        item = {"eeg":torch.from_numpy(x.copy()), "key":r.key, "index":index}
        if self.training_keys is not None:
            item["key"] = self.training_keys[index]
        return item

class DistinctSentenceBatches(Sampler):
    """Every observation is visited; rare singleton-class tails repeat a different anchor.

    This prevents zero-negative contrastive batches at microbatch=2. It does not
    increase the number of negatives via gradient accumulation.
    """
    def __init__(self, keys, batch_size, shuffle, seed=42):
        if batch_size < 2 or len(set(keys)) < 2:
            raise ValueError("Alignment requires batch_size>=2 and >=2 distinct sentences per split")
        self.keys, self.batch_size, self.shuffle, self.seed = keys, batch_size, shuffle, seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(len(self.keys), generator=generator).tolist() if self.shuffle else list(range(len(self.keys)))
        self.batches = self._batches(order)

    def _batches(self, order):
        pending = list(order)
        result = []
        while pending:
            first = pending.pop(0)
            match = next((i for i, j in enumerate(pending) if self.keys[j] != self.keys[first]), None)
            if match is None:
                second = next(j for j in order if self.keys[j] != self.keys[first])
            else:
                second = pending.pop(match)
            batch = [first, second]
            while pending and len(batch) < self.batch_size:
                batch.append(pending.pop(0))
            result.append(batch)
        return result

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def make_splits(records, config, protocol="subject", source=None, target=None):
    seed = config.get("seed", 42)
    if protocol in {"transfer", "transfer_exploratory"}:
        return transfer_split([r for r in records if r.dataset == source],
                              [r for r in records if r.dataset == target], seed,
                              require_identity_audit=protocol == "transfer")
    return group_split(records, seed, held_out_sentences=protocol == "subject_sentence")

def loader(records, config, normalizer=None, training=False, alignment=False, control="real", corruption=None):
    dataset = ObservationView(records, normalizer, config.get("max_samples", 1600), control, corruption, config.get("seed",42))
    batch = config.get("batch_size",2)
    options = dict(num_workers=config.get("num_workers",0), pin_memory=config.get("pin_memory",True) and torch.cuda.is_available(), collate_fn=collate_variable)
    if alignment:
        keys = dataset.training_keys or [r.key for r in records]
        batches = DistinctSentenceBatches(keys,batch,training,config.get("seed",42))
        return DataLoader(dataset,batch_sampler=batches,**options)
    return DataLoader(dataset,batch_size=batch,shuffle=training,**options)

def prepare_metadata(records, splits, config, stage, protocol):
    train = splits["train"]
    norm = ChannelNormalizer().fit(train, {r.subject_key for r in train})
    split = split_manifest(splits,config.get("seed",42),protocol)
    digest = hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
    metadata = {"config": config, "stage": stage, "channels": train[0].channels, "sfreq": train[0].sfreq,
                "normalizer": norm.state_dict(), "splits": split, "split_digest": digest,
                "candidate_keys": sorted({r.key for r in splits["test"]}),
                "train_datasets":sorted({r.dataset for r in splits["train"]}),
                "test_datasets":sorted({r.dataset for r in splits["test"]}),
                "research_data_status": "synthetic_fixture_no_scientific_claim" if any(r.provenance.get("synthetic") for r in records) else "user_supplied_not_independently_authenticated"}
    if protocol in {"transfer", "transfer_exploratory"}:
        metadata["transfer_assessment"] = compatibility(
            splits["train"] + splits["val"], splits["test"],
            require_identity_audit=protocol == "transfer")
        metadata["transfer_assessment"]["interpretation"] = (
            "Cross-release participant independence unresolved; do not claim unseen-person transfer"
            if protocol == "transfer_exploratory" else "Verified participant-disjoint transfer")
    return norm, metadata

def train_neural(records, config, stage, output, text_path=None, ssl_checkpoint=None, resume=None,
                 protocol="subject", source=None, target=None, control="real", features_path=None):
    if (Path(output)/"last.pt").exists() and not resume:
        raise FileExistsError("Training output already contains a checkpoint; use --resume or a new output directory")
    seed_all(config.get("seed",42))
    device, info = device_info(config.get("device","auto"))
    config = dict(config)
    config["batch_size"] = conservative_batch(config,info)
    splits = make_splits(records,config,protocol,source,target)
    norm, meta = prepare_metadata(records,splits,config,stage,protocol)
    meta["device_info"], meta["training_control"] = info, control
    train = loader(splits["train"],config,norm,True,stage == "alignment",control)
    val = loader(splits["val"],config,norm,False,stage == "alignment",
                 "gaussian" if control == "gaussian" else "real")
    Path(output).mkdir(parents=True,exist_ok=True)
    write_json(Path(output)/"splits.json",meta["splits"])
    write_json(Path(output)/"config.json",config)
    if stage == "ssl":
        return train_ssl(train,val,len(meta["channels"]),config,device,output,meta,resume)
    text = TextCache(text_path)
    for r in records:
        if r.key not in text.index:
            raise ValueError("Text cache missing a manifest sentence")
    meta["text_cache_digest"] = text.digest
    if features_path:
        if ssl_checkpoint:
            raise ValueError("Choose frozen LaBraM features or a ZuCo SSL encoder")
        from .pretrained_workflow import FeatureCache, FeatureIdentity, feature_loader, variant_name
        features = FeatureCache(features_path)
        features.verify(records, config.get("quality_aware", False), config.get("seed",42))
        matrix = features.matrix(splits["train"],variant_name(control))
        scaler = {"mean":matrix.mean(0).tolist(), "std":matrix.std(0).clip(1e-6).tolist()}
        meta.update(encoder_origin="cached_labram", feature_scaler=scaler,
                    feature_manifest_digest=features.metadata["manifest_digest"])
        train = feature_loader(splits["train"],features,config,control,True,scaler)
        val = feature_loader(splits["val"],features,config,"gaussian" if control == "gaussian" else "real",False,scaler)
        model = EEGTextModel(FeatureIdentity(),matrix.shape[1],text.embeddings.shape[1],frozen=True)
        history = train_alignment(model,train,val,text,config,device,output,meta,resume)
        return model,history
    encoder = EEGCNN(len(meta["channels"]),config["embedding_dim"],quality_aware=config.get("quality_aware",False))
    if ssl_checkpoint:
        ssl = MaskedEEG(len(meta["channels"]),config["embedding_dim"],config.get("quality_aware",False))
        previous = load_checkpoint(ssl_checkpoint,ssl)
        if previous["stage"] != "ssl" or previous["split_digest"] != meta["split_digest"] or previous["normalizer"] != meta["normalizer"]:
            raise ValueError("SSL checkpoint protocol/normalization differs; possible pretraining leakage")
        encoder = ssl.encoder
    meta["encoder_origin"] = "zuco_ssl" if ssl_checkpoint else "supervised_cnn"
    model = EEGTextModel(encoder,config["embedding_dim"],text.embeddings.shape[1])
    history = train_alignment(model,train,val,text,config,device,output,meta,resume)
    return model,history

def feature_matrix(records, config, norm, path, control="real"):
    dataset = ObservationView(records,norm,config.get("max_samples",1600),control=control,seed=config.get("seed",42))
    first = eeg_features(dataset[0]["eeg"].numpy(),records[0].sfreq)
    array = np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(records),len(first)))
    for i in range(len(records)):
        signal = dataset[i]["eeg"]
        if config.get("quality_aware",False):
            from .preprocessing.quality import channel_reliability
            signal = signal * channel_reliability(signal[None])[0]
        array[i] = eeg_features(signal.numpy(),records[i].sfreq)
    array.flush()
    return array

def train_classical(records, config, text_path, output, protocol="subject", source=None,target=None,control="real",features_path=None):
    output = Path(output)
    if (output/"metadata.json").exists():
        raise FileExistsError("Ridge output already exists; choose a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    splits = make_splits(records,config,protocol,source,target)
    norm, meta = prepare_metadata(records,splits,config,"ridge",protocol)
    text = TextCache(text_path)
    meta.update(text_cache_digest=text.digest,training_control=control)
    if features_path:
        from .pretrained_workflow import FeatureCache, variant_name
        features = FeatureCache(features_path)
        features.verify(records, config.get("quality_aware",False), config.get("seed",42))
        train = features.matrix(splits["train"],variant_name(control))
        val = features.matrix(splits["val"],variant_name(control))
        meta["encoder_origin"] = "cached_labram"
    else:
        train = feature_matrix(splits["train"],config,norm,output/"train_features.npy",control="gaussian" if control == "gaussian" else "real")
        val = feature_matrix(splits["val"],config,norm,output/"val_features.npy",control="gaussian" if control == "gaussian" else "real")
    train_keys = [r.key for r in splits["train"]]
    if control == "shuffled":
        train_keys = shuffled_keys(train_keys,config.get("seed",42))
    target_vectors = text.get(train_keys).astype(np.float64)
    pool = sorted({r.key for r in splits["val"]})
    best, chosen, table = -float("inf"), None, []
    for alpha in [.1,1.,10.,100.]:
        model = ridge_model(alpha).fit(train,target_vectors)
        metric = summarize(ranks(model.predict(val),text.get(pool),[r.key for r in splits["val"]],pool))["mrr"]
        table.append({"alpha":alpha,"validation_mrr":metric})
        if metric > best:
            best,chosen = metric,model
    scaler, ridge = chosen.named_steps["standardscaler"], chosen.named_steps["ridge"]
    np.savez_compressed(output/"ridge.npz",mean=scaler.mean_,scale=scaler.scale_,coef=ridge.coef_,intercept=ridge.intercept_)
    meta["selection"] = table
    meta.update(ridge_solver="svd", ridge_fit_dtype="float64", selected_alpha=float(ridge.alpha))
    write_json(output/"metadata.json",meta)
    write_json(output/"splits.json",meta["splits"])
    return table

def select_test(records, meta):
    identities = {tuple(v) for v in meta["splits"]["splits"]["test"]["observations"]}
    selected = [r for r in records if (r.dataset,r.subject,r.session,r.task,r.sentence_id) in identities]
    expected = meta["splits"]["splits"]["test"]["digest"]
    if manifest_digest(selected) != expected:
        raise ValueError("Test manifest or signals changed since training")
    return selected

@torch.no_grad()
def evaluate_checkpoint(records, artifact, text_path, output, control="auto", corruption=None, features_path=None):
    artifact = Path(artifact)
    ridge = artifact.suffix == ".npz"
    if ridge:
        meta = json.loads((artifact.parent/"metadata.json").read_text())
    else:
        meta = load_checkpoint(artifact)
        if meta["stage"] != "alignment":
            raise ValueError("Evaluate an aligned checkpoint, not raw SSL reconstruction")
    config = meta["config"]
    if control == "auto":
        control = "gaussian" if meta["training_control"] == "gaussian" else "real"
    text = TextCache(text_path)
    if text.digest != meta["text_cache_digest"]:
        raise ValueError("Sentence cache changed since model selection")
    selected = select_test(records,meta)
    norm = ChannelNormalizer.from_state(meta["normalizer"])
    device,_ = device_info(config.get("device","auto"))
    cached = meta.get("encoder_origin") == "cached_labram"
    if cached:
        if not features_path:
            raise ValueError("Supply --features for a frozen LaBraM artifact")
        from .pretrained_workflow import FeatureCache, FeatureIdentity, variant_name
        features = FeatureCache(features_path)
        features.verify(records,config.get("quality_aware",False), config.get("seed",42))
        matrix = features.matrix(selected,variant_name(control,corruption))
    if ridge:
        with np.load(artifact,allow_pickle=False) as f:
            coefficients = {k:f[k] for k in f.files}
        dataset = ObservationView(selected,norm,config.get("max_samples",1600),control,corruption,config.get("seed",42))
        predicted = []
        for i in range(len(dataset)):
            signal = dataset[i]["eeg"]
            if not cached and config.get("quality_aware",False):
                from .preprocessing.quality import channel_reliability
                signal = signal * channel_reliability(signal[None])[0]
            vector = matrix[i] if cached else eeg_features(signal.numpy(),selected[i].sfreq)
            scaled = (vector-coefficients["mean"])/coefficients["scale"]
            predicted.append(scaled @ coefficients["coef"].T + coefficients["intercept"])
        predicted = np.asarray(predicted)
    else:
        encoder = FeatureIdentity() if cached else EEGCNN(len(meta["channels"]),config["embedding_dim"],quality_aware=config.get("quality_aware",False))
        model = EEGTextModel(encoder,matrix.shape[1] if cached else config["embedding_dim"],text.embeddings.shape[1],frozen=cached)
        load_checkpoint(artifact,model)
        model.to(device).eval()
        predicted = []
        if cached:
            scaler = meta["feature_scaler"]
            for start in range(0,len(matrix),config.get("batch_size",2)):
                x = ((matrix[start:start+config.get("batch_size",2)]-np.array(scaler["mean"]))/np.array(scaler["std"])).astype(np.float32)
                predicted.append(model(torch.from_numpy(x)[:,:,None].to(device)).cpu().numpy())
        else:
            for b in loader(selected,config,norm,control=control,corruption=corruption):
                predicted.append(model(b["eeg"].to(device),b["mask"].to(device)).cpu().numpy())
        predicted = np.concatenate(predicted)
    pool = meta["candidate_keys"]
    query_keys = [r.key for r in selected]
    if control == "shuffled":
        query_keys = shuffled_keys(query_keys,config.get("seed",42))
    rank = ranks(predicted,text.get(pool),query_keys,pool)
    subjects = np.array([r.subject_key for r in selected])
    report = {"status":"executed", "artifact":str(artifact.resolve()), "protocol":meta["splits"]["protocol"],
              "data_status":meta["research_data_status"],
              "train_datasets":meta["train_datasets"], "test_datasets":meta["test_datasets"],
              "quality_aware":config.get("quality_aware",False), "seed":config.get("seed",42),
              "training_control":meta["training_control"], "evaluation_control":control,
              "corruption":corruption, "candidate_pool":"unique_test_sentences_fixed_per_checkpoint",
              "candidate_keys":pool,"candidate_pool_sha256":hashlib.sha256(json.dumps(pool).encode()).hexdigest(),
              "metrics":summarize(rank), "chance":chance(len(pool)),
              "subject_wise":{s:summarize(rank[subjects==s]) for s in sorted(set(subjects))},
              "macro_subject_top1_ci":subject_bootstrap((rank<=1).astype(float),subjects),
              "macro_subject_mrr_ci":subject_bootstrap(1/rank,subjects)}
    if "transfer_assessment" in meta:
        report["transfer_assessment"] = meta["transfer_assessment"]
    write_json(output,report)
    return report
