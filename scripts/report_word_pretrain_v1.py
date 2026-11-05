"""Render only saved training-CV measurements; never load EEG arrays or fit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from run_gru_retrieval_loss import ROOT, write


def read(path): return json.loads(path.read_text())


def main(output):
    dest=ROOT/'docs/research_results/word_pretrain_v1_20261101'
    if dest.exists(): raise FileExistsError('Fresh report directory required')
    assert read(output/'fit_complete.json')['complete'] and read(output/'evidence_verification.json')['passed']
    summary=read(output/'training_cv_summary.json');gaps=read(output/'real_minus_controls.json')
    decision=read(output/'decision.json');feas=read(output/'feasibility.json');plan=read(output/'plan.json')
    preservation=read(output/'preservation_after.json');cfg=plan['configuration']
    dest.mkdir(parents=True)
    lines=['# Exported-word supervision before sentence retrieval — v1', '',
        '**Training-reader CV only. Feasibility passed. Prespecified descriptive gate: '+str(decision['descriptive_gate_passed'])+'. Fixed run ended; no outer-validation or test evaluation.**', '',
        ('This bounded word-pretraining pilot did not establish a useful, consistent sentence-retrieval improvement beyond the fresh baselines and controls.' if not decision['descriptive_gate_passed'] else 'The descriptive training-only gate passed; this remains development evidence and does not authorize additional evaluation.'), '',
        '## Source feasibility and unchanged targets', '',
        f"All {feas['paired_observations']:,} original training observations and {feas['word_slots']:,} ordered word slots were checked directly against the eight training-reader source MAT files. For every selected slot, joining exported word.content with spaces exactly reproduced the unchanged parent sentence/manifest target. All eight-band × 105-channel TRT values and every missing position matched the saved word cache exactly. Source hashes match the prior audited sources. The exported word order is stimulus order, not chronological fixation order. No punctuation repair, lowercasing, regex filtering, token splitting or stimulus CSV substitution was applied.", '',
        'The documented mapping is in docs/word_eeg_feasibility_20261030.md, especially “Sentence-word and fixation mapping evidence.” The new opt-in bounded reader materializes only requested training slots and only parent content, exported word content and the eight TRT vectors. It skips raw waveforms, raw ET, gaze coordinates, durations, word bounds, fixations and all other fields. No outer-validation/test source file was opened. Trustworthy exported row/label pairing does not independently certify gaze assignment, physiological quality or physical sentence containment.', '',
        '## Sentence retrieval: fixed training-CV results', '',
        'Equal-reader macro scores over the original eight reader-disjoint holdout folds, 2,245 observations and unchanged 300-candidate pool. Recall is percent. MRR is not accuracy. Every sentence observation is retained, including all-missing ZMG/70. All scores use the final fixed checkpoint; no epoch or setting was selected from these outcomes.', '',
        '| Arm | Word updates | Sentence updates | R@1 % | R@5 % | R@10 % | MRR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    sentence_csv=[]
    for arm,row in summary.items():
        m=row['macro'];wu=128 if arm.startswith('pretrained_') else 0
        su=384 if arm=='fresh_equal_steps' else (0 if arm=='structural_only' else 256)
        lines.append(f"| {arm} | {wu} | {su} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |")
        for s,metrics in row['per_reader'].items(): sentence_csv.append({'arm':arm,'reader':s,**metrics})
    lines += ['', '## Pretrained real EEG minus comparators', '',
        '| Comparator | ΔR@1 pp | ΔR@5 pp | ΔR@10 pp | ΔMRR | Positive R@1 readers | Gate |',
        '|---|---:|---:|---:|---:|---:|---|']
    for arm,row in gaps.items():
        m=row['macro_difference'];positive=sum(d['recall@1']>0 for d in row['per_reader_difference'].values())
        lines.append(f"| {arm} | {100*m['recall@1']:+.4f} | {100*m['recall@5']:+.4f} | {100*m['recall@10']:+.4f} | {m['mrr']:+.6f} | {positive}/8 | {decision['checks'][arm]} |")
    lines += ['', 'Before fitting, the gate required pretrained real EEG to exceed every comparator by at least 2 percentage points Recall@1 and 0.005 MRR, with positive Recall@1 differences for at least six of eight readers against every comparator. This is a descriptive screen, not a significance test. The protocol stops after training CV regardless of the gate.', '',
        '## Word task: separate from sentence retrieval', '',
        'Word text is supervised target metadata only. Each fold builds an exact-string vocabulary from its seven fit readers only. The targets are fixed one-hot classes; no text embedding model, external vocabulary or target-side learned transform is used. Case and punctuation remain part of the class. Held-out labels never expand the vocabulary. Every held-out slot is retained; out-of-vocabulary targets receive a miss and reciprocal rank zero. Missing EEG slots also remain in both word training and evaluation, with zero imputed EEG plus an observed-word indicator. Word accuracy must not be read as sentence accuracy.', '',
        '| Arm | Macro word R@1 % | R@5 % | R@10 % | MRR |', '|---|---:|---:|---:|---:|']
    word_rows=[]
    for folder in sorted(output.glob('cv_*')):
        reader=folder.name[3:]
        for arm in cfg['arms']:
            if arm.startswith('pretrained_'):
                report=read(folder/arm/'word_holdout.json')
                word_rows.append({'arm':arm,'reader':reader,'vocabulary_size':report['vocabulary_size'],
                    'oov_slots':report['oov_slots'],'missing_eeg_slots':report['missing_eeg_slots'],
                    **report['all_word_slots']})
    for arm in cfg['arms']:
        rows=[r for r in word_rows if r['arm']==arm]
        if rows:
            m={k:sum(r[k] for r in rows)/len(rows) for k in ('recall@1','recall@5','recall@10','mrr')}
            lines.append(f"| {arm} | {100*m['recall@1']:.4f} | {100*m['recall@5']:.4f} | {100*m['recall@10']:.4f} | {m['mrr']:.6f} |")
    lines += ['', '| Holdout reader | Fold-training word classes | Held-out word slots | OOV slots | Missing EEG slots |',
        '|---|---:|---:|---:|---:|']
    for row in [r for r in word_rows if r['arm']=='pretrained_real']:
        lines.append(f"| {row['reader']} | {row['vocabulary_size']} | {row['n_queries']} | {row['oov_slots']} | {row['missing_eeg_slots']} |")
    lines += ['', 'Word metrics score all slots. Per-reader word results, observed-only and observed/in-vocabulary metrics, per-slot ranks, true target indices and missingness masks are saved separately in word_holdout.json and word_task_per_reader.csv. These reader folds repeat sentence classes; vocabulary coverage here must not be interpreted as evidence of unseen-word generalization. No word-level statistical independence or confidence interval is claimed. Frequent-word prediction can score above uniform chance without useful EEG information; matched controls are required.', '',
        '## Architecture, transfer and training budgets', '',
        'The unchanged sentence architecture has 45,952 parameters: 841→32 linear projection (26,944), tanh, one 32-unit GRU (6,336), and a 32→384 sentence-embedding head (12,672). The word encoder uses the same 841→32 projection and tanh with a new 32→V linear classification head. Its parameter count is 26,944 + 33V, where V is the fold-training vocabulary size, recorded for each fold. The only transferred tensors are project.weight (32×841) and project.bias (32), totaling 26,944 parameters. The word head is discarded. The GRU, sentence head and optimizer state do not transfer; GRU and sentence head retain the common original initialization. Every transfer was checked tensor by tensor.', '',
        'Numeric inputs contain only 840 source EEG coordinates and one observed-word indicator. No text, word IDs, sentence IDs, subject IDs, target vectors or eye-tracking measurements enter model input tensors. The sentence loss uses unchanged cached sentence embeddings only on its target/candidate side. EEG undergoes the existing signed log1p, per-word/per-band 105-channel centering and fold-training complete-word standardization. Missing slots become scaled training-mean zeros and remain recurrent steps. True word lengths, rather than valid-word counts, drive packed sequences; padded tails are excluded. All-missing observations retain their original length and learnable bias/length fallback.', '',
        'Seed 42, projection/hidden width 32, AdamW LR .001, weight decay .01, gradient clipping 1, word minibatches 256 and sentence minibatches up to 64 are fixed. The word stage runs exactly 128 optimizer updates with cross entropy over fold-training word classes. The sentence stage runs exactly 256 updates with full-candidate cosine-logit cross entropy at temperature .07. Batch orders use deterministic repeated full permutations with short end-of-cycle batches. Every update loss and batch size is saved. AdamW resets at the stage boundary. Only final stage checkpoints are scored; no early stopping or hyperparameter search occurred.', '',
        'The same-sentence fresh baseline trains a newly initialized GRU for 256 sentence updates without word training. The equal-step fresh baseline trains for 128 sentence updates, resets AdamW, then runs the same 256-update sentence stage, totaling 384 updates. All pretrained arms use 128 word + 256 sentence updates, also totaling 384. Matching by optimizer-update count does not match FLOPs, sequence exposure or number of supervised examples: a word update uses independent 841-coordinate slots, while a sentence update processes complete sequences and updates the GRU. Both baselines are reported to separate these budget effects; the historical selected-epoch GRU scores are context, not the fresh baseline.', '',
        'Gaussian, class-shuffled and missingness-only arms use exactly the same two stages, architecture, initial tensors, update budgets and optimizers. Gaussian draws preserve exact word counts and coordinate missingness; raw marginal moments fit fold-training words only, and Gaussian scaling fits Gaussian fold-training words. The Gaussian null does not preserve real covariance/full distribution. Class shuffling consistently deranges exact word classes during word training and sentence classes during sentence training, with correct held-out labels at both tasks; mappings are saved. This is a whole-supervision control. Missingness-only zeros all EEG amplitudes while preserving indicators and sentence lengths. Structural-only uses the same observations, first 14 count/duration summaries, fold-training count basis, training-fitted median/scaler and fixed ridge alpha 1. Its ranks exactly match the prior structural recipe on these folds.', '',
        '## Per-reader sentence retrieval', '',
        '| Arm | Reader | Queries | R@1 % | R@5 % | R@10 % | MRR |', '|---|---|---:|---:|---:|---:|---:|']
    for r in sentence_csv:
        lines.append(f"| {r['arm']} | {r['reader']} | {r['n_queries']} | {100*r['recall@1']:.4f} | {100*r['recall@5']:.4f} | {100*r['recall@10']:.4f} | {r['mrr']:.6f} |")
    lines += ['', '## Verification, preservation and limitations', '',
        f"Fourteen focused tests passed before fitting, covering source pairing, unchanged strings, training-only vocabulary/OOV/scaling, Gaussian masks, padding invariance, projection-only transfer, fixed budgets, loss and reader separation. Post-run replay verified all 48 final neural sentence models and all 32 word models, fold-training vocabularies, transferred tensors, fixed histories, per-reader ranks and aggregate metrics. Sentence initialization tensors match the earlier GRU exactly. {preservation['hashed_files']:,} prior protected files remain SHA256-identical; {preservation['dataset_metadata_files']:,} dataset/source metadata entries retain size and modification time. Existing local changes, datasets, targets, checkpoints, results, main report, presentation and .venv were preserved. No packages were installed or upgraded.", '',
        'These are development results on the same eight readers already used in prior experiments. Although no outcomes select epochs or settings in this run, its hypothesis and design were informed by development history. Overlapping folds, repeated words and repeated sentences are not independent cohorts. No significance claim, independent word-level sample size or confidence interval is justified. Previously reused outer validation would not be independent confirmation; neither outer-validation nor test EEG arrays were loaded or evaluated here. No test features were extracted. The task is retrieval among repeated known sentence classes, not unrestricted generation or unseen-sentence decoding.', '',
        'The source remains gaze/stimulus-assisted and annotation-segmented. Word order and missingness expose structure; word-stage one-hot supervision can learn frequent lexical classes without semantic generalization. This minimal stage uses one short budget, one seed and only a shared projection, so failure does not show that every form of word pretraining fails or that EEG contains no lexical information. Unverified exact producer revisions, source-session preprocessing, gamma2 descriptor/code discrepancy and possible physical boundary tails remain. No text repair, target/candidate/exclusion change or search expansion was made.', '',
        '## Artifacts and exact commands', '',
        'The new result directory contains the frozen configuration and implementation hashes; training source/cache/target hashes; exact exported target rows; source pairing evidence; candidate pools; fold identities and vocabularies; all preprocessing and Gaussian moments; class mappings; initial, word-stage and sentence-stage checkpoints; update histories; per-word and per-sentence ranks; all comparisons; decision; access record; replay verification; preservation checks; and command/log records. It contains no outer/test metrics.', '',
        '```powershell',r'.\.venv\Scripts\python.exe -m pytest tests/test_word_pretrain_v1.py tests/test_gru_pilot.py tests/test_gru_retrieval_loss.py -q',
        r'.\.venv\Scripts\python.exe scripts/prepare_word_pretrain_v1.py --output results/word_pretrain_v1_20261101',
        r'.\.venv\Scripts\python.exe -u scripts/run_word_pretrain_v1.py --output results/word_pretrain_v1_20261101',
        r'.\.venv\Scripts\python.exe -u scripts/verify_word_pretrain_v1.py --output results/word_pretrain_v1_20261101',
        r'.\.venv\Scripts\python.exe scripts/report_word_pretrain_v1.py --output results/word_pretrain_v1_20261101','```','']
    (dest/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    for name,rows in [('sentence_per_reader.csv',sentence_csv),('word_task_per_reader.csv',word_rows)]:
        with (dest/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name in ('plan.json','feasibility.json','training_cv_summary.json','real_minus_controls.json','decision.json',
                 'environment.json','commands.json','evidence_verification.json','preservation_after.json','access_summary.json'):
        shutil.copyfile(output/name,dest/name)
    write(output/'word_task_summary.json',word_rows);shutil.copyfile(output/'word_task_summary.json',dest/'word_task_summary.json')
    for source in (Path(__file__),ROOT/'scripts/verify_word_pretrain_v1.py'):
        shutil.copyfile(source,output/'implementation'/source.name)
    write(output/'implementation_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/'implementation').iterdir()})
    shutil.copyfile(output/'implementation_sha256.json',dest/'implementation_sha256.json')
    print(dest/'README.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
