"""Prepare shared transfer metadata/text targets without training or inventing identities."""
import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import read_manifest, write_manifest
from .validation import validate_records, manifest_digest
from ..evaluation.cross_dataset import compatibility
from ..training.alignment_train import TextCache
from ..utils.logging import write_json


def prepare_plan(manifest1, manifest2, text1, text2, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new transfer output directory; existing files are preserved')
    cohorts = {'zuco1':read_manifest(manifest1), 'zuco2':read_manifest(manifest2)}
    validation = {}
    for version, records in cohorts.items():
        if {r.dataset for r in records} != {version}:
            raise ValueError(f'{version} manifest contains an unexpected dataset version')
        print(f'Checking {version}: {len(records)} observations', flush=True)
        validation[version] = validate_records(records)
        if not validation[version]['valid']:
            raise ValueError(f'{version} validation failed: {validation[version]["errors"][:3]}')
        if any(r.provenance.get('units')!='uV' for r in records):
            raise ValueError('Transfer inputs require documented physical microvolt units')
    # This technical check is explicitly not an identity audit or training permission.
    technical = compatibility(cohorts['zuco1'], cohorts['zuco2'], require_identity_audit=False)
    caches = [TextCache(text1), TextCache(text2)]
    fields = ['model','requested_revision','resolved_revision','sentence_hash_rule']
    if not caches[0].metadata.get('model') or any(
            caches[0].metadata.get(k) != caches[1].metadata.get(k) for k in fields):
        raise ValueError('Text model/revision/hash conventions differ; build one shared frozen text cache')
    if caches[0].embeddings.shape[1] != caches[1].embeddings.shape[1]:
        raise ValueError('Text embedding dimensions differ')
    vectors = {}
    for cache in caches:
        for key, vector in zip(cache.keys, cache.embeddings):
            if key in vectors and not np.allclose(vectors[key],vector,rtol=1e-6,atol=1e-7):
                raise ValueError('Shared sentence embeddings disagree across source caches')
            vectors.setdefault(key,vector)
    records = cohorts['zuco1'] + cohorts['zuco2']
    keys = sorted({r.key for r in records})
    if not set(keys) <= vectors.keys():
        raise ValueError('Sentence embeddings missing from the supplied caches')
    directions = []
    for source,target in [('zuco1','zuco2'),('zuco2','zuco1')]:
        a,b = cohorts[source],cohorts[target]
        seen = {r.key for r in a}
        try:
            compatibility(a,b)
            identity_passes, reason = True, None
        except ValueError as exc:
            identity_passes, reason = False,str(exc)
        directions.append(dict(source=source,target=target,status='not_executed',
                               identity_gate_passes=identity_passes,blocking_reason=reason,
                               target_observations=len(b),target_candidates=len({r.key for r in b}),
                               target_observations_with_source_unseen_text=sum(r.key not in seen for r in b),
                               target_candidates_with_source_unseen_text=len({r.key for r in b}-seen),
                               rule='Fit EEG normalization/SSL/projection/ridge and select hyperparameters on source only'))
    output.mkdir(parents=True)
    write_manifest(output/'combined.jsonl',records)
    metadata = {k:caches[0].metadata.get(k) for k in fields}
    metadata.update(source_cache_digests=[c.digest for c in caches],
                    construction='Union of frozen cached embeddings; shared vectors checked within numeric tolerance',
                    identity_evidence='Not established by text-cache construction')
    np.savez_compressed(output/'sentence_embeddings.npz',keys=np.array(keys),
                        embeddings=np.stack([vectors[k] for k in keys]).astype(np.float32),
                        metadata=json.dumps(metadata))
    report = dict(status='prepared_not_executed' if all(d['identity_gate_passes'] for d in directions)
                  else 'blocked_pending_participant_identity',
                  manifest=str(output/'combined.jsonl'),text_cache=str(output/'sentence_embeddings.npz'),
                  directions=directions,technical_compatibility=technical,
                  n_observations=len(records),n_unique_sentences=len(keys),
                  cohort_digests={k:manifest_digest(v) for k,v in cohorts.items()},
                  n_subjects={k:len({r.subject for r in v}) for k,v in cohorts.items()},
                  identity_policy='No IDs assigned or changed; genuine supporting evidence must be reviewed before strict execution',
                  experiment_config='configs/transfer_ridge_only.yaml',
                  input_config='configs/labram_full_sentence.yaml')
    write_json(output/'plan.json',report)
    print(f'Plan: {output/"plan.json"}\nStatus: {report["status"]}',flush=True)
    print(f'{len(records)} references, {len(keys)} sentence targets; no EEG copies and no training',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zuco1',required=True)
    parser.add_argument('--zuco2',required=True)
    parser.add_argument('--text-zuco1',required=True)
    parser.add_argument('--text-zuco2',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    prepare_plan(args.zuco1,args.zuco2,args.text_zuco1,args.text_zuco2,args.output)


if __name__=='__main__':
    main()
