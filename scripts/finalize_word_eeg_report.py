"""Fill measured feasibility tables and preserve a new audit archive; no training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/word_eeg_feasibility_20261030'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive=ROOT/'docs/research_results/word_eeg_feasibility_20261030'
    report=ROOT/'docs/word_eeg_feasibility_20261030.md'
    if archive.exists():raise FileExistsError(archive)
    s=json.loads((BASE/'summary/summary.json').read_text());fields=json.loads((BASE/'summary/field_census.json').read_text())
    d=json.loads((BASE/'summary/derived_details.json').read_text())
    text=report.read_text(encoding='utf-8')
    assert all(token in text for token in ('<!-- MEASURED_COVERAGE -->','<!-- WORD_COUNT_COLLISIONS -->','<!-- STORAGE_AND_VALIDATION -->'))
    coverage=['The full census is complete. All 30 source MAT hashes match the pinned local download plans. '
              'Here, “usable” means a fully finite `(105,N)` fixation with `N≥2`, or eight fully finite '
              '105-channel vectors for a selected reading-measure family. It does not establish physiological '
              'quality, semantic information or valid all-run timing.\n\n',
              '| Coverage | ZuCo1 | ZuCo2 |\n|---|---:|---:|\n']
    for label,key,scope in [('Source sentence slots','sentence_observations','source_slots'),('Slots with word records','with_word_records','source_slots'),
          ('Word observations, repeated across readers','word_observations','source_slots'),('Slots with ≥1 usable raw fixation / TRT word','at_least_one_complete_TRT_word','source_slots'),
          ('Existing prepared sentence observations','sentence_observations','prepared'),('Existing duration-QC observations','sentence_observations','duration_qc'),
          ('QC observations with ≥1 usable raw fixation / TRT word','at_least_one_complete_TRT_word','duration_qc')]:
        coverage.append(f"| {label} | {s['zuco1'][scope][key]:,} | {s['zuco2'][scope][key]:,} |\n")
    coverage.append('\nThere are **100 ZuCo1 and 199 ZuCo2 slots without word records**. Among all slots that '
        'do have word records, joining the words reproduces the parent sentence text exactly in '
        '**3,500/3,500** ZuCo1 and **6,083/6,083** ZuCo2 observations; their word-bound row counts also match. '
        'Thus the 299 absent-word slots are missingness, not observed contradictory text mappings. All existing '
        'prepared/QC targets match their source sentence text. Word observations repeat sentences and lexical '
        'items across readers; they are not unique word types or independent cohorts.\n\n')
    coverage += ['### Word-feature missingness across all exported word records\n\n',
        'Each cell is **finite 8×105 / empty / all-NaN vector / absent field**. The same state counts were '
        'verified for all eight bands within each family; finite counts require all eight bands, not just theta. '
        'Denominators are 76,891 ZuCo1 and 119,271 ZuCo2 word observations.\n\n',
        '| Family | ZuCo1 | ZuCo2 |\n|---|---:|---:|\n']
    for m in ('FFD','SFD','GD','GPT','TRT'):
        cells=[' / '.join(f"{d[v]['per_band_same_missingness_counts'][m].get(k,0):,}" for k in ('finite','empty','all_nan_vector','absent')) for v in ('zuco1','zuco2')]
        coverage.append(f"| {m} | {cells[0]} | {cells[1]} |\n")
    coverage.append('\nRaw fixation cells contain **89,643 finite matrices and 306 scalar-NaN cells** in ZuCo1, '
        'and **122,213 finite matrices and 597 scalar-NaN cells** in ZuCo2. Finite snippet lengths range '
        'from **51–1,018 samples** and **51–1,293 samples**, respectively. No partly nonfinite raw matrix '
        'or partly nonfinite band vector was observed in this census. `rawEEG` itself is absent from '
        '**205 / 1,351** word records, rather than merely empty. Valid ZuCo1 raw cells are float32; '
        'ZuCo2 raw cells are float64. The spectral vectors are float64 when present. Sampled `rawET` '
        'cells are `(4,N)` in both releases; auxiliary row identities and a separate rawET cell-value '
        'quality census are not established here.\n\n')
    coverage += ['### Fixed ZuCo1 seed-42 retrieval split\n\n',
        '| Split | Existing observations | With usable word EEG | Without usable word EEG | Complete TRT word observations |\n|---|---:|---:|---:|---:|\n']
    for name in ('train','val','test'):
        c=s['zuco1_saved_seed42_splits'][name];n=c['sentence_observations'];good=c['at_least_one_complete_TRT_word']
        coverage.append(f"| {name} | {n:,} | {good:,} | {n-good} | {c['TRT_complete_8x105_words']:,} |\n")
    coverage.append('\nAll seven are word records with no fixated words and omission rate 1, not absent sentence '
        'targets: training **ZMG/70**; test **ZKB/71, 72, 81, 85, 93, 94** (zero-based sentence IDs). '
        'No new exclusions were made. In ZuCo2, 34 QC observations have no usable word EEG; their exact '
        'identities are in `derived_details.json`. The QC denominators remain 3,349 and 5,884.\n\n'
        'Word features also exist for **144 ZuCo1 and 168 ZuCo2 observations outside the existing prepared '
        'sentence-EEG cohorts**. They may support a separately reviewed word-view cohort, but cannot be '
        'silently added to an architecture comparison. They remain outside the current datasets.\n\n')
    text=text.replace('<!-- MEASURED_COVERAGE -->',''.join(coverage))
    collision=s['word_count_candidate_collisions']['validation']
    text=text.replace('<!-- WORD_COUNT_COLLISIONS -->',
        f"The static metadata audit makes the length shortcut concrete: the **{collision['n_candidates']} validation candidates** "
        f"have only **{collision['distinct_word_counts']} distinct word counts**, ranging from **6 to 65**. "
        '**Eight sentence classes are unique by word count alone**. If exact displayed word count is exposed, '
        f"the largest remaining equal-count candidate bucket has only **{collision['largest_equal_count_bucket']} sentences**. "
        'The training pool has 300 candidates with the same 47 counts and eight unique-count classes. '
        'These are collision counts, not fitted-model retrieval scores or an accuracy estimate.\n')
    # Raw payload lower bounds; no feature dataset is actually created.
    words=sum(s[v]['source_slots']['word_observations'] for v in ('zuco1','zuco2'))
    qcobs=sum(s[v]['duration_qc']['sentence_observations'] for v in ('zuco1','zuco2'))
    rawbytes=0
    for v in ('zuco1','zuco2'):
        for shape,n in fields[v]['raw_fixation_fields']['rawEEG_fixation']['shapes'].items():
            dims=json.loads(shape)
            if len(dims)==2 and dims[0]==105:rawbytes+=n*dims[0]*dims[1]*4
    estimated={'all_source_word_slots':words,'single_840_feature_family_float32_bytes':words*840*4,
               'qc_sentence_pooled_840_float32_bytes':qcobs*840*4,'all_finite_raw_fixations_float32_bytes':rawbytes,
               'scope':'Uncompressed array payload estimates only; no feature dataset written'}
    (BASE/'summary/storage_estimate.json').write_text(json.dumps(estimated,indent=2)+'\n',encoding='utf-8')
    ending=f'''A float32 840-feature family for all {words:,} exported word slots has an uncompressed
array payload of about **{words*840*4/2**30:.2f} GiB**, before masks, metadata and indexing.
Pooling the {qcobs:,} existing QC observations to 840 values each would require about
**{qcobs*840*4/2**20:.1f} MiB**. The finite raw-fixation samples would require approximately
**{rawbytes/2**30:.2f} GiB** if converted to float32. These are storage arithmetic,
not a new cache, measured model memory usage or training-time benchmarks.

## Reproduction, preservation and audit limits

The audit used the existing `.venv`. **5 focused tests passed in 0.65 s** for
bounded nested Level-5 structs/cells, HDF5 axis/empty conventions, missingness,
scalar-struct handling and exact-containment rejection. No training tests or
model fits ran. All **30 MAT source hashes** match the pinned download plans;
the actual source files were opened read-only and their sizes/mtimes remained
unchanged. All **494 protected prior files** (results/checkpoints, manifests and
text caches) retained their hashes. Existing dirty worktree changes were preserved.

Commands for the completed census/consolidation included:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests/test_word_eeg_audit.py -q
.\\.venv\\Scripts\\python.exe -u scripts/audit_word_eeg.py --dataset zuco1 --output results/word_eeg_feasibility_20261030/zuco1
.\\.venv\\Scripts\\python.exe -u scripts/audit_word_eeg.py --dataset zuco2 --output results/word_eeg_feasibility_20261030/zuco2
.\\.venv\\Scripts\\python.exe -u scripts/audit_word_eeg.py --dataset zuco2 --workers 4 --completed results/word_eeg_feasibility_20261030/zuco2 --output results/word_eeg_feasibility_20261030/zuco2_parallel
.\\.venv\\Scripts\\python.exe scripts/probe_word_eeg_containment.py
.\\.venv\\Scripts\\python.exe scripts/summarize_word_eeg_audit.py
```

The initial sequential ZuCo2 audit was stopped to partition the remaining reads
among four bounded worker processes. Its completed YAC report and log were
preserved and counted once; incomplete work on YAG was not treated as a result.
The Level-5 reader was not changed semantically when parallel orchestration was
added. Source sentence `rawData` matrix values were skipped in the full word
census; only the ten-cell containment probe read its two parent waveforms.
Word/fixation EEG finite values and all named word bands were inspected fully.
Shape/finiteness does not verify original gaze classification, artifact removal,
run-specific reference, calibration or every acquisition-clock mapping.

The [audit archive](research_results/word_eeg_feasibility_20261030/README.md)
contains the full field census, sentence coverage, exact all-empty identities,
source hashes, channel and pair definitions, waveform probe, source readers and
code/configuration provenance. Per-subject inspection reports remain under
`results/word_eeg_feasibility_20261030/`. Output guards require fresh paths; no
old experiment file was patched or retargeted.
'''
    text=text.replace('<!-- STORAGE_AND_VALIDATION -->',ending)
    report.write_text(text,encoding='utf-8')
    archive.mkdir(parents=True)
    sources=[*sorted((BASE/'summary').glob('*.json')),*sorted(BASE.glob('osf*.json')),
        BASE/'getElectrodePairs.m',BASE/'data_loading_helpers.py',BASE/'read_matlab_files.py',
        BASE/'containment_examples.json',BASE/'containment_followup.json',BASE/'preservation_before.json',
        ROOT/'results/pairing_audit_20261030/official_firstLevel_NR.m',ROOT/'results/pairing_audit_20261030/osf_matlab_inventory.json',
        ROOT/'scripts/audit_word_eeg.py',ROOT/'scripts/probe_word_eeg_containment.py',ROOT/'scripts/summarize_word_eeg_audit.py',
        ROOT/'scripts/finalize_word_eeg_report.py',ROOT/'tests/test_word_eeg_audit.py',
        BASE/'implementation/audit_word_eeg_initial.py',
        *[ROOT/'logs'/n for n in ('word_eeg_zuco1_20261030.log','word_eeg_zuco2_20261030.log','word_eeg_zuco2_parallel_20261030.log')]]
    manifest=[]
    for source in sources:
        folder='implementation' if source.suffix=='.py' else 'logs' if source.suffix=='.log' else 'evidence'
        target=archive/folder/source.name;target.parent.mkdir(exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(source,target);assert sha(source)==sha(target)
        manifest.append({'source':str(source.relative_to(ROOT)),'copy':str(target.relative_to(ROOT)),'sha256':sha(source)})
    subject_reports=[]
    for directory in ('zuco1','zuco2','zuco2_parallel'):
        for p in (BASE/directory).glob('*.json'):
            if p.name=='progress.json':continue
            j=json.loads(p.read_text(encoding='utf-8'))
            subject_reports.append({'dataset':j['dataset'],'subject':j['subject'],'audit_report':str(p.relative_to(ROOT)),
                'audit_report_sha256':sha(p),'counts':j['counts'],'examples':j['examples']})
    assert len(subject_reports)==30
    (archive/'per_subject_counts_and_examples.json').write_text(json.dumps(subject_reports,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (archive/'archive_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (archive/'README.md').write_text('# Word/fixation EEG feasibility audit\n\n'
        'See the [full report](../../word_eeg_feasibility_20261030.md).\n\n'
        'All 12 ZuCo1 and 18 ZuCo2 NR sentence exports were inspected read-only. '
        'The evidence supports a separately versioned gaze-assisted sentence-retrieval view, with '
        'explicit missingness, boundary and structural-leakage controls. It does not establish reliable '
        'EEG retrieval or complete recording-clock validity. No datasets/targets changed and no models trained.\n\n'
        '`evidence/summary.json`, `field_census.json`, `sentence_coverage.json` and `derived_details.json` '
        'hold measured coverage, shape/state counts and exact all-empty QC identities. '
        '`source_inventory.json` pins all 30 source MAT hashes; `channel_schemas.json` and '
        '`electrode_pairs.json` preserve channel definitions. The containment evidence includes one '
        'fixation extending 39 samples beyond its parent sentence array. '
        '`per_subject_counts_and_examples.json` preserves inspection examples and hashes of all '
        'full per-subject reports. `archive_manifest.json` verifies source/copy hashes. '
        'The original sequential ZuCo2 partial audit is preserved; remaining subjects were audited with '
        'four bounded reader processes, with no duplicate subject counted.\n\n'
        'Five audit tests passed. All 494 protected prior artifact/input files remained unchanged. '
        'Implementation estimate: 5–8 engineering days for a controlled spectral view; '
        '4–7 additional days for a verified raw-fixation path, subject to source availability.\n',encoding='utf-8')
    print(json.dumps({'report':str(report),'archive':str(archive),'archive_copies':len(manifest),'storage':estimated},indent=2))


if __name__=='__main__':main()
