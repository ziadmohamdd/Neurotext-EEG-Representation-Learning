"""SYNTHETIC end-to-end execution. Metrics here only test the software."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from eeg_decoding.workflow import train_neural,train_classical,evaluate_checkpoint,DistinctSentenceBatches
from eeg_decoding.training.checkpoints import load_checkpoint
from eeg_decoding.training.trainer import Trainer
from eeg_decoding.models.eeg_cnn import EEGCNN
from eeg_decoding.utils.config import load_config
from eeg_decoding.cli import main
from eeg_decoding.data.dataset import write_manifest


def test_baseline_end_to_end(records,config,text_cache,tmp_path):
    output=tmp_path/"ridge"
    train_classical(records,config,text_cache.path,output)
    for control in ["real","gaussian","shuffled"]:
        report=evaluate_checkpoint(records,output/"ridge.npz",text_cache.path,tmp_path/(control+".json"),control)
        assert report["metrics"]["n_queries"]==10
        assert report["data_status"]=="synthetic_fixture_no_scientific_claim"
    report=evaluate_checkpoint(records,output/"ridge.npz",text_cache.path,tmp_path/"corruption.json",corruption=("temporal_mask",.25))
    assert len(report["candidate_keys"])==10


def test_ssl_alignment_checkpoint_evaluation(records,config,text_cache,tmp_path):
    ssl_output=tmp_path/"ssl"
    _,history=train_neural(records,config,"ssl",ssl_output)
    assert len(history)==2 and (ssl_output/"best.pt").is_file()
    output=tmp_path/"aligned"
    _,history=train_neural(records,config,"alignment",output,text_cache.path,ssl_output/"best.pt")
    assert len(history)==2
    report=evaluate_checkpoint(records,output/"best.pt",text_cache.path,tmp_path/"metrics.json")
    assert report["metrics"]["n_queries"]==10
    assert report["candidate_pool"]=="unique_test_sentences_fixed_per_checkpoint"
    # Resuming a completed schedule performs no extra epochs and preserves artifacts.
    _,resumed=train_neural(records,config,"alignment",output,text_cache.path,ssl_output/"best.pt",output/"last.pt")
    assert len(resumed)==2
    different=dict(config,seed=43)
    with pytest.raises(ValueError,match="SSL checkpoint protocol"):
        train_neural(records,different,"alignment",tmp_path/"bad",text_cache.path,ssl_output/"best.pt")


def test_exact_interrupted_resume(config,tmp_path):
    from eeg_decoding.utils.reproducibility import seed_all
    seed_all(123)
    x=torch.randn(2,3,100)
    mask=torch.ones(2,100,dtype=torch.bool)
    batches=[{"eeg":x,"mask":mask}]*4
    objective=lambda model,batch: model(batch["eeg"],batch["mask"]).square().mean()
    seed_all(456)
    full=EEGCNN(3)
    trainer=Trainer(full,config,torch.device("cpu"),tmp_path/"full")
    trainer.fit(batches,batches,objective,{"config":config})
    seed_all(456)
    interrupted=EEGCNN(3)
    calls=[0]
    def fail_after_epoch(model,batch):
        calls[0]+=1
        if calls[0]>8:
            raise RuntimeError("simulated interruption")
        return objective(model,batch)
    with pytest.raises(RuntimeError,match="simulated"):
        Trainer(interrupted,config,torch.device("cpu"),tmp_path/"resume").fit(batches,batches,fail_after_epoch,{"config":config})
    restarted=EEGCNN(3)
    Trainer(restarted,config,torch.device("cpu"),tmp_path/"resume").fit(batches,batches,objective,{"config":config},tmp_path/"resume"/"last.pt")
    for a,b in zip(full.parameters(),restarted.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_distinct_sentence_batches():
    keys=["a"]*7+["b"]*2+["c"]
    for size in [2,3,8]:
        sampler=DistinctSentenceBatches(keys,size,False)
        batches=list(sampler)
        assert set(range(len(keys)))<=set(sum(batches,[]))
        assert all(len({keys[i] for i in b})>=2 for b in batches)
        assert all(len(b)<=size for b in batches)


def test_cli_ingest_preprocess(records,config,tmp_path):
    import yaml
    manifest=tmp_path/"input.jsonl"
    write_manifest(manifest,records[:10])
    prep=dict(config,preprocessing={"target_sfreq":200,"low":.5,"high":45,"notch":50,
        "already_filtered":False,"artifact_threshold_uv":200,"artifact_policy":"reject"})
    path=tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(prep))
    output=tmp_path/"clean.jsonl"
    main(["preprocess","--manifest",str(manifest),"--config",str(path),"--cache",str(tmp_path/"cache"),"--output",str(output)])
    assert output.is_file()
    main(["inspect-data","--manifest",str(output),"--output",str(tmp_path/"validation.json")])
    assert json.loads((tmp_path/"validation.json").read_text())["valid"]


def test_feature_alignment_and_ridge(records,config,text_cache,tmp_path):
    from eeg_decoding.data.validation import manifest_digest
    from eeg_decoding.pretrained_workflow import observation_id
    feature_dir=tmp_path/"SYNTHETIC_features"
    feature_dir.mkdir()
    rng=np.random.default_rng(9)
    for name in ["real","gaussian","additive_noise_0.5"]:
        np.save(feature_dir/(name+".npy"),rng.normal(size=(len(records),200)).astype(np.float32))
    (feature_dir/"metadata.json").write_text(json.dumps({"status":"complete","variants":["real","gaussian","additive_noise_0.5"],
          "manifest_digest":manifest_digest(records),"observation_ids":[observation_id(r) for r in records],"quality_aware":False,"seed":42}))
    for stage in ["ridge","alignment"]:
        output=tmp_path/stage
        if stage=="ridge":
            train_classical(records,config,text_cache.path,output,features_path=feature_dir)
            artifact=output/"ridge.npz"
        else:
            train_neural(records,config,"alignment",output,text_cache.path,features_path=feature_dir)
            artifact=output/"best.pt"
        result=evaluate_checkpoint(records,artifact,text_cache.path,output/"eval.json",features_path=feature_dir)
        assert result["metrics"]["n_queries"]==10
        result=evaluate_checkpoint(records,artifact,text_cache.path,output/"noise.json",features_path=feature_dir,corruption=("additive_noise",.5))
        assert result["corruption"]==("additive_noise",.5)


@pytest.mark.parametrize("control",["gaussian","shuffled"])
@pytest.mark.parametrize("quality",[False,True])
def test_separate_training_controls(records,config,text_cache,tmp_path,control,quality):
    config=dict(config,epochs=1,quality_aware=quality)
    train_classical(records,config,text_cache.path,tmp_path/"ridge_control",control=control)
    train_neural(records,config,"alignment",tmp_path/"cnn_control",text_cache.path,control=control)
    report=evaluate_checkpoint(records,tmp_path/"cnn_control/best.pt",text_cache.path,tmp_path/"eval.json")
    assert report["training_control"]==control


def test_cross_dataset_train_both_directions(records,config,text_cache,tmp_path):
    from dataclasses import replace
    source=[r for r in records if int(r.subject)<3]
    target=[replace(r,dataset="zuco2") for r in records if int(r.subject)>=3]
    both=source+target
    for src,dst in [("zuco1","zuco2"),("zuco2","zuco1")]:
        output=tmp_path/(src+"_to_"+dst)
        train_classical(both,config,text_cache.path,output,protocol="transfer",source=src,target=dst)
        report=evaluate_checkpoint(both,output/"ridge.npz",text_cache.path,output/"eval.json")
        assert report["protocol"]=="transfer" and report["metrics"]["n_queries"]==30


def test_experiment_runner_executes_and_exports(records,config,text_cache,tmp_path):
    import yaml
    manifest=tmp_path/"records.jsonl"
    write_manifest(manifest,records)
    cfg=tmp_path/"config.yaml"
    cfg.write_text(yaml.safe_dump(dict(config,epochs=1)))
    spec=tmp_path/"experiments.yaml"
    spec.write_text(yaml.safe_dump({"seeds":[42],"representations":["ridge","cnn","ssl"],
        "protocols":["held_out_subject"],"controls":["real"],"quality_aware":[False],"robustness":{"temporal_mask":[0,.25]}}))
    output=tmp_path/"runs"
    main(["run-experiments","--manifest",str(manifest),"--config",str(cfg),"--experiments",str(spec),
          "--text-cache",str(text_cache.path),"--output",str(output),"--execute"])
    jobs=json.loads((output/"jobs.json").read_text())
    assert len(jobs)==3 and all(j["status"]=="executed" for j in jobs)
    assert (output/"retrieval.csv").is_file()
