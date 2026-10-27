"""Command-line entry point shared by scripts, notebooks, Windows and Colab."""
import argparse
import csv
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from .data.dataset import read_manifest, write_manifest, load_signal
from .data.validation import validate_records, manifest_digest
from .utils.config import load_config
from .utils.logging import write_json
from .utils.device import device_info

def common(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config",default="configs/local.yaml")

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    sub.add_parser("device")
    inspect = sub.add_parser("inspect-data")
    inspect.add_argument("--manifest")
    inspect.add_argument("--root")
    inspect.add_argument("--output",default="results/metrics/data_validation.json")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--format",choices=["bids","mat"],required=True)
    ingest.add_argument("--dataset",choices=["zuco1","zuco2"],required=True)
    ingest.add_argument("--root",required=True)
    ingest.add_argument("--alignment",help="Verified sentence-clock alignment TSV for BIDS")
    ingest.add_argument("--schema",help="Explicit metadata JSON for one MATLAB recording")
    ingest.add_argument("--skip-missing-eeg",action="store_true",help="Exclude scalar NaN/empty MATLAB EEG slots, preserving original sentence IDs and writing an import audit")
    ingest.add_argument("--cache",default="data/processed/imported")
    ingest.add_argument("--output",required=True)
    prep = sub.add_parser("preprocess")
    common(prep)
    prep.add_argument("--output",required=True)
    prep.add_argument("--cache",default="data/processed/clean")
    prep.add_argument("--channels",help="JSON list, explicit shared channel order")
    text = sub.add_parser("cache-text")
    common(text)
    text.add_argument("--output",required=True)
    pretrained = sub.add_parser("extract-pretrained")
    common(pretrained)
    pretrained.add_argument("--cache",default="checkpoints/official_labram")
    pretrained.add_argument("--channel-mapping",required=True)
    pretrained.add_argument("--output",required=True)
    pretrained.add_argument("--include-controls",action="store_true")
    pretrained.add_argument("--seed",type=int,help="Override extraction/control seed")
    for name in ["train-baseline","train-ssl","train-alignment"]:
        train = sub.add_parser(name)
        common(train)
        train.add_argument("--output",required=True)
        train.add_argument("--text-cache")
        train.add_argument("--resume")
        train.add_argument("--ssl-checkpoint")
        train.add_argument("--features",help="Frozen LaBraM cache for ridge or contrastive projection")
        train.add_argument("--protocol",choices=["subject","subject_sentence","transfer"],default="subject")
        train.add_argument("--source",choices=["zuco1","zuco2"])
        train.add_argument("--target",choices=["zuco1","zuco2"])
        train.add_argument("--control",choices=["real","gaussian","shuffled"],default="real")
        train.add_argument("--quality-aware",action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--manifest",required=True)
    evaluate.add_argument("--artifact",required=True)
    evaluate.add_argument("--text-cache",required=True)
    evaluate.add_argument("--features")
    evaluate.add_argument("--output",required=True)
    evaluate.add_argument("--control",choices=["auto","real","gaussian","shuffled"],default="auto")
    evaluate.add_argument("--robustness",action="store_true")
    experiments = sub.add_parser("run-experiments")
    experiments.add_argument("--manifest")
    experiments.add_argument("--text-cache")
    experiments.add_argument("--features")
    experiments.add_argument("--quality-features")
    experiments.add_argument("--config",default="configs/local.yaml")
    experiments.add_argument("--experiments",default="configs/experiments.yaml")
    experiments.add_argument("--output",default="results/experiment_runs")
    experiments.add_argument("--execute",action="store_true",help="Without this flag, only save not-executed jobs")
    export = sub.add_parser("export-results")
    export.add_argument("--root",default="results")
    export.add_argument("--output",default="results/tables/retrieval.csv")
    return p

def inspect_data(args):
    if args.manifest:
        report = validate_records(read_manifest(args.manifest))
        write_json(args.output,report)
        if not report["valid"]:
            raise ValueError(f"Invalid data; see {args.output}")
        print(f"Validated {report['n_observations']} observations, {len(report['subjects'])} subjects")
    elif args.root:
        root = Path(args.root)
        paths = sorted(root.rglob("*_eeg.json"))
        report = {"recordings": [{"path":str(path), "metadata":json.loads(path.read_text())} for path in paths],
                  "mat_files":[str(p) for p in root.rglob("*.mat")],
                  "alignment_status":"not_verified; raw events do not imply sentence alignment"}
        write_json(args.output,report)
        print(f"Inspected {len(paths)} BIDS sidecars; see {args.output}")
    else:
        raise ValueError("Supply --manifest or --root")

def ingest(args):
    audit = {}
    if args.format == "bids":
        if not args.alignment:
            raise ValueError("BIDS requires --alignment; trigger-only tables are insufficient")
        from .data.nemar import iter_bids
        records = iter_bids(args.root,args.alignment,args.dataset,args.cache)
    else:
        if not args.schema:
            raise ValueError("MATLAB requires inspected --schema JSON")
        from .data.zuco1 import iter_mat as load1
        from .data.zuco2 import iter_mat as load2
        schema = json.loads(Path(args.schema).read_text())
        records = (load1 if args.dataset == "zuco1" else load2)(args.root,schema,args.cache,
                    skip_missing_eeg=args.skip_missing_eeg, audit=audit)
    try:
        write_manifest(args.output,records)
    except Exception as exc:
        if args.format == "mat":
            write_json(str(args.output)+".import.json",dict(audit,status="failed",error=str(exc)))
        raise
    if args.format == "mat":
        write_json(str(args.output)+".import.json",dict(audit,status="imported" if audit.get("imported") else "empty"))
    report = validate_records(read_manifest(args.output))
    write_json(str(args.output)+".validation.json",report)
    if not report["valid"]:
        raise ValueError("Imported data failed validation")
    print(f"Imported {report['n_observations']} real-source observations; scientific provenance must still be audited")

def preprocess(args):
    from .preprocessing.filtering import filter_resample
    from .preprocessing.artifacts import handle_artifacts
    from .preprocessing.quality import signal_quality
    from .preprocessing.segmentation import harmonize
    from .utils.plotting import signal_plot
    config = load_config(args.config)["preprocessing"]
    channels = json.loads(Path(args.channels).read_text()) if args.channels else None
    cache = Path(args.cache).resolve()
    cache.mkdir(parents=True,exist_ok=True)
    records = read_manifest(args.manifest)
    kept, rejected = [], []
    for r in records:
        previous = r.provenance.get("preprocessing")
        if previous:
            if previous == config and (channels is None or r.channels == channels):
                kept.append(r)
                continue
            raise ValueError("Input already preprocessed differently; use the original imported manifest to avoid double filtering")
        x = load_signal(r)
        if channels:
            x = harmonize(x,r.channels,channels)
        quality_before = signal_quality(x)
        try:
            filtered = filter_resample(x,r.sfreq,**{k:v for k,v in config.items() if k not in {"artifact_policy","artifact_threshold_uv"}})
            x = handle_artifacts(filtered,config["artifact_threshold_uv"],config["artifact_policy"])
        except ValueError as exc:
            rejected.append({"dataset":r.dataset,"subject":r.subject,"sentence_id":r.sentence_id,"reason":str(exc)})
            continue
        key = hashlib.sha256((manifest_digest([r])+json.dumps(config,sort_keys=True)+json.dumps(channels)).encode()).hexdigest()
        dest = cache / f"{key}.npz"
        np.savez_compressed(dest,eeg=x)
        provenance = dict(r.provenance,preprocessing=config,previous_source=r.signal_path,units="uV")
        kept.append(replace(r,signal_path=str(dest),channels=channels or r.channels,sfreq=config["target_sfreq"],
                            quality={"before":quality_before,"after":signal_quality(x)},provenance=provenance))
        if len(kept) == 1:
            signal_plot(x,config["target_sfreq"],cache/"diagnostic.png")
    write_manifest(args.output,kept)
    write_json(str(args.output)+".report.json",{"kept":len(kept),"rejected":rejected,"config":config})
    if not kept:
        raise ValueError("All observations rejected; inspect preprocessing report")
    print(f"Preprocessed {len(kept)} observations; rejected {len(rejected)}")

def train(args):
    from .workflow import train_neural, train_classical
    records, config = read_manifest(args.manifest), load_config(args.config)
    config["quality_aware"] = args.quality_aware or config.get("quality_aware",False)
    if args.command != "train-ssl" and not args.text_cache:
        raise ValueError("--text-cache is required")
    if args.command == "train-baseline":
        result = train_classical(records,config,args.text_cache,args.output,args.protocol,args.source,args.target,args.control,args.features)
    else:
        result = train_neural(records,config,"ssl" if args.command == "train-ssl" else "alignment",args.output,
                              args.text_cache,args.ssl_checkpoint,args.resume,args.protocol,args.source,args.target,args.control,args.features)
    print(f"Training completed; artifacts: {args.output}")

def evaluate(args):
    from .workflow import evaluate_checkpoint
    from .utils.plotting import robustness_plot
    records = read_manifest(args.manifest)
    report = evaluate_checkpoint(records,args.artifact,args.text_cache,args.output,args.control,features_path=args.features)
    print(json.dumps(report["metrics"],indent=2))
    if args.robustness:
        rows = []
        path = Path(args.output)
        for kind,levels in [("additive_noise",[0,.25,.5,1.]),("channel_dropout",[0,.1,.25,.5]),("temporal_mask",[0,.1,.25,.5])]:
            for severity in levels:
                result = evaluate_checkpoint(records,args.artifact,args.text_cache,
                            path.with_name(f"{path.stem}_{kind}_{severity}.json"),
                            corruption=(kind,severity),features_path=args.features)
                rows.append(dict(corruption=kind,severity=severity,**result["metrics"]))
        write_json(path.with_name(path.stem+"_robustness.json"),rows)
        robustness_plot(rows,path.with_name(path.stem+"_robustness.png"))

def run_experiments(args):
    spec = load_config(args.experiments)
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    jobs = []
    for seed,representation,protocol,control,quality in itertools.product(spec["seeds"],spec["representations"],spec["protocols"],spec["controls"],spec["quality_aware"]):
        job = dict(seed=seed,representation=representation,protocol=protocol,control=control,quality_aware=quality,status="not_executed")
        jobs.append(job)
    write_json(root/"jobs.json",jobs)
    if not args.execute:
        print(f"Saved {len(jobs)} not-executed experiment specifications. Real data, text cache and LaBraM caches required for execution.")
        return
    if not args.manifest or not args.text_cache:
        raise ValueError("Execution requires --manifest and --text-cache")
    if "labram" in spec["representations"] and (not args.features or not args.quality_features):
        raise ValueError("LaBraM ablations require --features and --quality-features generated with --include-controls")
    from .workflow import train_neural,train_classical,evaluate_checkpoint
    records = read_manifest(args.manifest)
    base = load_config(args.config)
    for i,job in enumerate(jobs):
        output = root/f"{i:04d}_{job['representation']}_{job['protocol']}_{job['control']}_q{int(job['quality_aware'])}_s{job['seed']}"
        config = dict(base,seed=job["seed"],quality_aware=job["quality_aware"])
        protocol = job["protocol"]
        source,target = None,None
        if "_to_" in protocol:
            source,target = protocol.split("_to_")
            protocol = "transfer"
        else:
            protocol = "subject_sentence" if protocol.endswith("and_sentence") else "subject"
        template = (args.quality_features if job["quality_aware"] else args.features)
        features = template.format(seed=job["seed"]) if job["representation"] == "labram" else None
        try:
            if job["representation"] == "ridge":
                train_classical(records,config,args.text_cache,output,protocol,source,target,job["control"])
                artifact = output/"ridge.npz"
            else:
                ssl = None
                if job["representation"] == "ssl":
                    train_neural(records,config,"ssl",output/"ssl",protocol=protocol,source=source,target=target,control=job["control"])
                    ssl = output/"ssl"/"best.pt"
                train_neural(records,config,"alignment",output,args.text_cache,ssl,protocol=protocol,source=source,target=target,control=job["control"],features_path=features)
                artifact = output/"best.pt"
            evaluate_checkpoint(records,artifact,args.text_cache,output/"evaluation.json",features_path=features)
            if job["control"] != "gaussian":
                for kind, levels in spec["robustness"].items():
                    for severity in levels:
                        evaluate_checkpoint(records,artifact,args.text_cache,output/f"robustness_{kind}_{severity}.json",corruption=(kind,severity),features_path=features)
                job["robustness_status"] = "executed"
            else:
                job["robustness_status"] = "not_requested_for_gaussian_training_control"
            job["status"] = "executed"
        except Exception as exc:
            job.update(status="failed",error=str(exc))
            write_json(root/"jobs.json",jobs)
            raise
        write_json(root/"jobs.json",jobs)
    export_results(argparse.Namespace(root=str(root),output=str(root/"retrieval.csv")))

def export_results(args):
    rows = []
    for path in sorted(Path(args.root).rglob("*.json")):
        report = json.loads(path.read_text())
        if isinstance(report,dict) and report.get("status") == "executed" and "metrics" in report:
            rows.append({"file":str(path),"data_status":report["data_status"],"protocol":report["protocol"],
                         "train_datasets":str(report["train_datasets"]),"test_datasets":str(report["test_datasets"]),
                         "quality_aware":report["quality_aware"],"seed":report["seed"],"training_control":report["training_control"],
                         "evaluation_control":report["evaluation_control"],"corruption":str(report["corruption"]),
                         "candidate_pool_sha256":report["candidate_pool_sha256"],**report["metrics"]})
    if not rows:
        raise ValueError("No executed evaluation reports found; no result table fabricated")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} executed evaluations")

def main(argv=None):
    args = parser().parse_args(argv)
    commands = {"inspect-data":inspect_data,"ingest":ingest,"preprocess":preprocess,"evaluate":evaluate,
                "run-experiments":run_experiments,"export-results":export_results}
    if args.command == "device":
        print(json.dumps(device_info()[1],indent=2))
    elif args.command == "cache-text":
        from .training.alignment_train import TextCache
        config = load_config(args.config)
        TextCache.build(read_manifest(args.manifest),args.output,config["text_model"],config.get("text_revision"))
    elif args.command == "extract-pretrained":
        from .pretrained_workflow import extract_features
        extract_features(args)
    elif args.command.startswith("train-"):
        train(args)
    else:
        commands[args.command](args)

if __name__ == "__main__":
    main()
