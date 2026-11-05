"""Report stored GRU pilot evidence; never fit or load EEG arrays."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/gru_pilot_v1_20261101'
DEST=ROOT/'docs/research_results/gru_pilot_v1_20261101'


def read(name): return json.loads((RUN/name).read_text())


def main():
    if DEST.exists(): raise FileExistsError('Preserve prior report')
    summary=read('training_cv_summary.json'); gaps=read('real_minus_controls.json')
    decision=read('decision.json'); preservation=read('preservation_after.json')
    selection=read('frozen_selection.json'); access=read('access_summary.json')
    assert preservation['unchanged'] and not access['test_arrays_loaded'] and not access['test_evaluated']
    arch=read('cv_ZDM/architecture.json'); plan=read('plan.json')
    DEST.mkdir(parents=True)
    lines=['# Full-word-sequence GRU pilot v1 — training-reader CV', '',
        '**One bounded pilot completed. Training-only descriptive gate: '+('passed' if decision['training_gate_passed'] else 'failed; stopped without outer validation')+'.**', '',
        'The real sequence encoder did not establish a useful, consistent advantage beyond the matched controls and structural comparator.' if not decision['training_gate_passed'] else 'The training-only gate permitted one frozen outer evaluation; see its separate artifact.', '',
        '## Training-reader holdout results', '',
        'These are equal-reader macro training-CV results on the original eight training readers, 2,245 observations and unchanged 300-candidate pool. Recall is percent; MRR is not accuracy. The common epoch was selected from 4/8/12 using real-EEG training-CV Recall@1, with ties preferring the earlier epoch. Selected epoch: **'+str(selection['common_epoch'])+'**. Every neural condition uses that same epoch.', '',
        '| Condition | R@1 % | R@5 % | R@10 % | MRR |', '|---|---:|---:|---:|---:|']
    for c,row in summary.items():
        m=row['macro']; lines.append(f"| {c} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |")
    lines += ['', '## Real EEG minus comparators', '',
        '| Comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Gate |',
        '|---|---:|---:|---:|---:|---:|---|']
    for c,row in gaps.items():
        g=row['macro_difference']; positive=sum(d['recall@1']>0 for d in row['per_reader_difference'].values())
        lines.append(f"| {c} | {100*g['recall@1']:+.4f} | {100*g['recall@5']:+.4f} | {100*g['recall@10']:+.4f} | {g['mrr']:+.6f} | {positive}/8 | {decision['checks'][c]} |")
    lines += ['', 'The prespecified gate requires real EEG to exceed every comparator by at least 2 percentage points Recall@1 and 0.005 MRR, with positive Recall@1 differences on at least six of eight readers for every comparison. It is a descriptive stopping rule, not a significance test.', '',
        '## Architecture, inputs and controls', '',
        f"The model has {arch['parameters']:,} trainable parameters: an 841→32 linear projection ({arch['layers']['projection']:,}), tanh, one unidirectional 32-unit GRU ({arch['layers']['gru']:,}), and a 32→384 sentence-embedding head ({arch['layers']['head']:,}). It consumes all source word positions with no binning, truncation or timing assumptions. Source word order is stimulus order, not chronological fixation order.", '',
        'The 840 EEG coordinates undergo signed log1p and within-word centering separately over 105 channels in each of eight bands. Mean/SD are fitted only on complete words from fold-training readers. Missing words are imputed to that training mean, producing scaled zeros, and an extra observed-word indicator is appended. Missing slots remain recurrent steps. Batch padding is separate: packed sequences use true word counts, so padded tails never reach the GRU. All-missing ZMG/70 remains as its full-length zero/indicator-zero sequence; predictions may reflect length and learned biases. A hypothetical zero-length sequence uses one zero fallback step. No word text, word ID, sentence ID, subject ID or eye-tracking measurement enters a neural input.', '',
        'Matched Gaussian draws preserve exact word counts and every missing coordinate. Their marginal moments are fitted on raw fold-training words only, before the shared nonlinear transform; Gaussian scaling is independently fitted on Gaussian fold-training words. This null does not preserve the full real-feature distribution or covariance. Word-order shuffling permutes complete vectors only among observed slots, preserving missing positions and sentence means. The mean comparator repeats each sentence’s mean scaled EEG vector at its observed slots, preserving count and mask. Missingness-only zeros all 840 amplitude coordinates while retaining the same indicator and true sequence length. Class shuffling consistently deranges training sentence classes across repeated observations; held-out labels remain true.', '',
        'All six neural conditions use identical architecture, seed-42 initial weights, minibatch order, AdamW (learning rate 0.001, weight decay 0.01), cosine embedding loss, batch size 64, gradient clipping 1, and 12 epochs with checkpoints at 4/8/12. No learning-rate, width or optimizer search was performed. The existing structural comparator uses the same observations and targets, its original 14 count/duration summaries plus a fold-training count basis, training-fitted median imputation/standardization and fixed ridge alpha 1. That alpha came from preceding training-only semantic CV; no new structural grid was searched. Duration summaries enter only this comparator.', '',
        '## Per-reader training-CV metrics', '',
        '| Condition | Reader | R@1 % | R@5 % | R@10 % | MRR |', '|---|---|---:|---:|---:|---:|']
    csv_rows=[]
    for c,row in summary.items():
        for s,m in row['per_reader'].items():
            lines.append(f"| {c} | {s} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |")
            csv_rows.append({'condition':c,'reader':s,**m})
    lines += ['', '## Outer validation and test status', '',
        ('Outer validation was not evaluated and its EEG cache members were not loaded because the training-only gate failed. No outer-validation scores are reported.' if not access['outer_evaluated'] else 'One frozen outer-validation evaluation is stored separately in outer_validation_summary.json; these are reused development readers, not independent confirmation.'), '',
        'No raw source MAT file was opened, no test feature array was loaded, and no test extraction or evaluation was performed. The existing 512-query/298-candidate outer cohort and all targets/exclusions remain unchanged.', '',
        '## Verification, preservation and limitations', '',
        f"Before fitting, 16 focused tests passed (5 new GRU tests and 11 existing word-order/spectral tests). They cover missingness, all-missing fallback, padding invariance, training-only transforms and moments, matched controls, class-consistent shuffling and reader separation. The existing .venv was used unchanged (Python 3.11.9, PyTorch 2.7.1+cu118, RTX 3050 Laptop GPU). The sandbox could not resolve its base interpreter; execution outside the sandbox succeeded without installation or upgrades. {preservation['hashed_files']} pre-existing files remain SHA256-identical; {preservation['raw_metadata_files']} dataset files retain size and modification time. Raw/test sources were inventoried through filesystem metadata only. Existing datasets, targets, checkpoints, results, report and local changes were preserved.", '',
        'These CV scores reuse the training readers both for duration selection and description and are optimistic development results. Prior experiments have already informed the representation and protocol. Outer validation has also been reused historically and would not provide independent confirmation. Repeated sentences and overlapping folds are not independent cohorts; no confidence interval or significance claim is made. Retrieval of known candidate sentences is not unrestricted text generation or unseen-sentence decoding. Gaze/stimulus-assisted segmentation, missingness and length can reveal annotation structure. Producer preprocessing, exact revision, gamma2 descriptor/code discrepancy and physical boundary limitations from the source audits remain. One small architecture, one seed and a short budget cannot establish that all sequence models fail or that EEG contains no semantic information. No search was expanded after this result.', '',
        '## Artifacts and commands', '',
        'The result directory contains the frozen plan/configuration, source/cache and implementation hashes, candidate pools, every fold identity and class mapping, train-only preprocessing and Gaussian moments, identical initial weights, checkpoints at every prespecified epoch, training histories, per-query ranks, all scored epochs, shared epoch selection, per-reader metrics, control differences, decision, access record and preservation checks. No previous report or result was replaced.', '',
        '```powershell', "$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'", r'.\.venv\Scripts\python.exe -m pytest tests/test_gru_pilot.py tests/test_word_order.py tests/test_spectral_pilot.py -q',
        r'.\.venv\Scripts\python.exe scripts/run_gru_pilot.py --output results/gru_pilot_v1_20261101',
        r'.\.venv\Scripts\python.exe scripts/report_gru_pilot.py', '```', '']
    (DEST/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    with (DEST/'training_cv_per_reader.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    for name in ['plan.json','training_cv_summary.json','real_minus_controls.json','decision.json','frozen_selection.json','preservation_after.json','access_summary.json','environment.json','coverage.json','commands.json','evidence_verification.json']:
        shutil.copyfile(RUN/name,DEST/name)
    hashes={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUN/'implementation').iterdir()}
    hashes['training_candidate_pool']=hashlib.sha256(json.dumps(plan['training_candidate_keys']).encode()).hexdigest()
    (DEST/'implementation_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(DEST/'README.md')


if __name__=='__main__': main()
