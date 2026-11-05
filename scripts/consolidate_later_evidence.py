"""Append later-study slides 27-35 to a copy of the 26-slide deck from saved evidence.

Reads saved CSV/JSON only. It fits, replays and evaluates nothing, and it refuses
existing destinations. Slides 1-26 keep their exact source bytes.

  build:  --output deck.pptx
  verify: --output deck.pptx --pdf deck.pdf --manifest manifest.json
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as E

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'docs/research_results'
REVIEW = RESEARCH/'evidence_review_v1_20261101'
SOURCE_DECK = ROOT/'docs/presentation/NeuroText_consolidated.pptx'
SOURCE_PDF = ROOT/'docs/presentation/NeuroText_consolidated.pdf'
SOURCE_DECK_MD5 = '65d0440fb8e70c19e6ca9fec7d4be8d9'
SOURCE_PDF_MD5 = '681413ecd10141fb8363e247a0048a7b'
SOURCES = [REVIEW/'training_cv.csv', REVIEW/'outer_validation.csv', REVIEW/'word_task_macro.json',
           REVIEW/'word_task.csv', REVIEW/'participant_metrics.csv', RESEARCH/'retrieval.csv',
           RESEARCH/'nway_word_pretrain_v1_20261101/summary.csv',
           RESEARCH/'amplitude_audit_v1_20261101/training_summary.json',
           RESEARCH/'gru_retrieval_loss_v1_20261101/baseline_prediction_audit/summary.json',
           RESEARCH/'gru_retrieval_loss_v1_20261101/prediction_concentration.json']
FOOTER = 'NEUROTEXT  /  REVIEW DRAFT  /  05 NOV 2026'
NAMESPACES = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
TEAL, INK, MUTED, TINT, BAND, AMBER = '087F8C', '142D44', '526474', 'E8F5F4', 'F1F5F8', 'A04A00'
LEFT, WIDTH = 502920, 11155680
TABLE_TOP, TABLE_BOTTOM = 1600200, 5620000


def rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def percent(fraction, places=3):
    return f'{100*float(fraction):.{places}f}%'


def score(fraction):
    return f'{float(fraction):.6f}'


def load_evidence():
    evidence = {'cv': {(r['study'], r['arm']): r for r in rows(REVIEW/'training_cv.csv')},
                'outer': {(r['study'], r['arm']): r for r in rows(REVIEW/'outer_validation.csv')},
                'nway': {(int(r['N']), r['arm']): r
                         for r in rows(RESEARCH/'nway_word_pretrain_v1_20261101/summary.csv')},
                'word': {r['arm']: r['metrics'] for r in
                         json.loads((REVIEW/'word_task_macro.json').read_text(encoding='utf-8'))}}
    return evidence


def positive_readers(reports, left, right, metric='recall@1'):
    assert left in reports and right in reports and set(reports[left]) == set(reports[right])
    return sum(float(reports[left][k][metric]) > float(reports[right][k][metric]) for k in reports[left])


def claims():
    """Recompute every number that slide prose states outside a generated table."""
    participant = {}
    for r in rows(REVIEW/'participant_metrics.csv'):
        participant.setdefault((r['scope'], r['study'], r['arm']), {})[r['reader']] = r
    cv = lambda study, arm: ('training_cv', study, arm)
    word = {}
    for r in rows(REVIEW/'word_task.csv'):
        word.setdefault(r['arm'], {})[r['reader']] = r
    outer = {r['arm']: r for r in rows(REVIEW/'outer_validation.csv') if r['study'] == 'word_order'}
    amplitude = json.loads((RESEARCH/'amplitude_audit_v1_20261101/training_summary.json').read_text(encoding='utf-8'))
    baseline = json.loads((RESEARCH/'gru_retrieval_loss_v1_20261101/baseline_prediction_audit/summary.json').read_text(encoding='utf-8'))
    concentration = json.loads((RESEARCH/'gru_retrieval_loss_v1_20261101/prediction_concentration.json').read_text(encoding='utf-8'))
    cross_entropy = [r['unique_top1_sentences'] for r in concentration if r['condition'] == 'real_eeg']
    # Corruption labels hold [kind, severity]; severity 0 is the clean reference.
    nway = rows(RESEARCH/'nway_word_pretrain_v1_20261101/summary.csv')
    robustness = {}
    for r in rows(RESEARCH/'retrieval.csv'):
        if r['corruption'] in ('', 'None') or not re.split(r'[\\/]', r['file'])[-1].startswith('zuco2_ridge'):
            continue
        robustness.setdefault(r['corruption'], {})[r['quality_aware']] = float(r['mrr'])
    nonzero = {k: v for k, v in robustness.items() if not k.rstrip(']').endswith(', 0')}
    corrupted = {re.split(r'[\\/]', r['file'])[-1].split('_')[1] for r in rows(RESEARCH/'retrieval.csv')
                 if r['corruption'] not in ('', 'None')}
    return {
        'ordered_minus_mean_top1_hits': round((float(outer['ordered_real']['recall@1'])-float(outer['mean_real']['recall@1']))*int(outer['ordered_real']['n_queries'])),
        'controls_above_ordered_recall1_and_mrr': all(float(outer[arm][m]) > float(outer['ordered_real'][m])
                                                      for arm in ('matched_gaussian', 'word_order_shuffled') for m in ('recall@1', 'mrr')),
        'word_real_near_chance_note_missingness_above_at_every_n': all(
            float(n['expected_accuracy']) < float(next(o['expected_accuracy'] for o in nway if o['N'] == n['N'] and o['arm'] == 'pretrained_missingness'))
            for n in nway if n['arm'] == 'pretrained_real'),
        'amplitude_combined_real_weights': sorted({a['selection']['weight'] for a in amplitude if a['condition'] == 'structural_real'}),
        'cosine_distinct_top1': baseline['real_eeg']['unique_top1_sentences'],
        'cross_entropy_distinct_top1': statistics.mean(cross_entropy),
        'gru_cosine_real_over_gaussian': positive_readers(participant, cv('gru', 'real_eeg'), cv('gru', 'matched_gaussian')),
        'gru_ce_real_over_gaussian': positive_readers(participant, cv('gru_ce', 'real_eeg'), cv('gru_ce', 'matched_gaussian')),
        'pretrained_over_fresh_same': positive_readers(participant, cv('word_pretrain', 'pretrained_real'), cv('word_pretrain', 'fresh_same_sentence')),
        'pretrained_over_missingness': positive_readers(participant, cv('word_pretrain', 'pretrained_real'), cv('word_pretrain', 'pretrained_missingness')),
        'word_real_over_missingness': positive_readers(word, 'pretrained_real', 'pretrained_missingness'),
        'quality_nonzero_conditions': len(nonzero),
        'quality_lower_conditions': sum(v['True'] < v['False'] for v in nonzero.values()),
        'models_with_corrupted_rows': sorted(corrupted),
    }


EXPECTED_CLAIMS = {'ordered_minus_mean_top1_hits': 1, 'controls_above_ordered_recall1_and_mrr': True,
                   'word_real_near_chance_note_missingness_above_at_every_n': True, 'amplitude_combined_real_weights': [0.0],
                   'cosine_distinct_top1': 6.375, 'cross_entropy_distinct_top1': 53,
                   'gru_cosine_real_over_gaussian': 3, 'gru_ce_real_over_gaussian': 3,
                   'pretrained_over_fresh_same': 0, 'pretrained_over_missingness': 0,
                   'word_real_over_missingness': 1, 'quality_nonzero_conditions': 9,
                   'quality_lower_conditions': 8, 'models_with_corrupted_rows': ['ridge']}


def run(text, size, color, bold=False):
    return (f'<a:r><a:rPr lang="en-US" sz="{size}" b="{int(bold)}"><a:solidFill><a:srgbClr val="{color}" /></a:solidFill>'
            f'<a:latin typeface="Aptos" /></a:rPr><a:t>{escape(text)}</a:t></a:r>')


def text_box(shape_id, x, y, w, h, paragraphs, anchor='t', after=0):
    body = ''.join(f'<a:p><a:pPr><a:spcAft><a:spcPts val="{after}" /></a:spcAft></a:pPr>{p}</a:p>' for p in paragraphs)
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id-1}" /><p:cNvSpPr txBox="1" /><p:nvPr /></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}" /><a:ext cx="{w}" cy="{h}" /></a:xfrm><a:prstGeom prst="rect"><a:avLst /></a:prstGeom><a:noFill /></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="18288" rIns="18288" tIns="18288" bIns="18288" anchor="{anchor}"><a:noAutofit /></a:bodyPr><a:lstStyle />{body}</p:txBody></p:sp>')


def rectangle(shape_id, x, y, w, h, color):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="Rectangle {shape_id-1}" /><p:cNvSpPr /><p:nvPr /></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}" /><a:ext cx="{w}" cy="{h}" /></a:xfrm><a:prstGeom prst="rect"><a:avLst /></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{color}" /></a:solidFill><a:ln><a:noFill /></a:ln></p:spPr>'
            '<p:style><a:lnRef idx="1"><a:schemeClr val="accent1" /></a:lnRef><a:fillRef idx="3"><a:schemeClr val="accent1" /></a:fillRef>'
            '<a:effectRef idx="2"><a:schemeClr val="accent1" /></a:effectRef><a:fontRef idx="minor"><a:schemeClr val="lt1" /></a:fontRef></p:style>'
            '<p:txBody><a:bodyPr rtlCol="0" anchor="ctr" /><a:lstStyle /><a:p><a:pPr algn="ctr" /></a:p></p:txBody></p:sp>')


def table(shape_id, table_rows, widths, size):
    assert sum(widths) == 10789920
    height = (TABLE_BOTTOM-TABLE_TOP)//len(table_rows)
    grid = ''.join(f'<a:gridCol w="{w}" />' for w in widths)
    body = ''
    for index, values in enumerate(table_rows):
        fill = TEAL if index == 0 else BAND if index % 2 else 'FFFFFF'
        cells = ''
        for value in values:
            color, bold = ('FFFFFF', True) if index == 0 else (TEAL, True) if value == 'Done' else (AMBER, True) if value == 'Partial' else (INK, False)
            cells += (f'<a:tc><a:txBody><a:bodyPr /><a:lstStyle /><a:p><a:pPr algn="l" />{run(str(value), size, color, bold)}</a:p></a:txBody>'
                      f'<a:tcPr marL="91440" marR="54864" marT="36576" marB="27432" anchor="ctr"><a:solidFill><a:srgbClr val="{fill}" /></a:solidFill></a:tcPr></a:tc>')
        body += f'<a:tr h="{height}">{cells}</a:tr>'
    return (f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{shape_id}" name="Results table" /><p:cNvGraphicFramePr /><p:nvPr /></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="685800" y="{TABLE_TOP}" /><a:ext cx="10789920" cy="{height*len(table_rows)}" /></p:xfrm>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            f'<a:tbl><a:tblPr firstRow="1" bandRow="1" /><a:tblGrid>{grid}</a:tblGrid>{body}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')


def slide_xml(number, title, subtitle, content, callout):
    shapes = [rectangle(2, 0, 0, 12191695, 109728, TEAL),
              text_box(3, LEFT, 320040, WIDTH, 594360, [run(title, 3000, INK, True)]),
              text_box(4, LEFT, 960120, WIDTH, 566928, [run(subtitle, 1600, MUTED)], anchor='ctr'),
              text_box(5, LEFT, 6492240, 10607040, 228600, [run(FOOTER, 1000, MUTED)]),
              text_box(6, 11064240, 6492240, 640080, 228600, [run(str(number), 1000, MUTED)])]
    shapes += content
    shapes += [rectangle(30, LEFT, 5779008, WIDTH, 530352, TINT),
               text_box(31, 667512, 5779008, 10789920, 530352, [run(callout, 1600, TEAL, True)], anchor='ctr')]
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:sld {NAMESPACES}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF" /></a:solidFill><a:effectLst /></p:bgPr></p:bg>'
            '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name="" /><p:cNvGrpSpPr /><p:nvPr /></p:nvGrpSpPr><p:grpSpPr />'
            f'{"".join(shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping /></p:clrMapOvr></p:sld>').encode('utf-8')


def notes_xml(text):
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<p:notes {NAMESPACES}><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name="" /><p:cNvGrpSpPr /><p:nvPr /></p:nvGrpSpPr>'
            '<p:grpSpPr><a:xfrm><a:off x="0" y="0" /><a:ext cx="0" cy="0" /><a:chOff x="0" y="0" /><a:chExt cx="0" cy="0" /></a:xfrm></p:grpSpPr>'
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1" /><p:cNvSpPr><a:spLocks noGrp="1" /></p:cNvSpPr><p:nvPr><p:ph type="sldImg" idx="2" /></p:nvPr></p:nvSpPr><p:spPr /></p:sp>'
            '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2" /><p:cNvSpPr><a:spLocks noGrp="1" /></p:cNvSpPr><p:nvPr><p:ph type="body" idx="3" sz="quarter" /></p:nvPr></p:nvSpPr><p:spPr />'
            f'<p:txBody><a:bodyPr /><a:lstStyle /><a:p><a:r><a:t>{escape(text)}</a:t></a:r></a:p></p:txBody></p:sp>'
            '<p:sp><p:nvSpPr><p:cNvPr id="4" name="Slide Number Placeholder 3" /><p:cNvSpPr><a:spLocks noGrp="1" /></p:cNvSpPr><p:nvPr><p:ph type="sldNum" idx="5" sz="quarter" /></p:nvPr></p:nvSpPr><p:spPr /></p:sp>'
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping /></p:clrMapOvr></p:notes>').encode('utf-8')


def metric_rows(source, study, arms):
    table_rows = [['Condition', 'R@1', 'R@5', 'R@10', 'MRR']]
    for label, arm in arms:
        r = source[study, arm]
        table_rows.append([label, percent(r['recall@1']), percent(r['recall@5']), percent(r['recall@10']), score(r['mrr'])])
    return table_rows


def slides(evidence):
    cv, outer, nway, word = evidence['cv'], evidence['outer'], evidence['nway'], evidence['word']
    metric_widths = [3840480, 1737360, 1737360, 1737360, 1737360]
    cv_scope = 'TRAINING-READER CV ONLY  /  8 readers  /  2,245 observations  /  300 candidates'
    deck = [dict(title='Later pilots: what changed after 30 October',
                 subtitle='SAVED RESULTS ONLY  /  No training or evaluation ran for this update  /  No test-set use',
                 table=[['Study', 'Change tested', 'Scope', 'Supported finding'],
                        ['Ordered-word ridge', 'Four word-position bins vs. mean', 'CV, then reused validation', 'One extra hit; controls stronger'],
                        ['Relative amplitude', 'Channel-relative features, two heads', 'Training CV', 'Zero EEG weight selected'],
                        ['Full-sequence GRU', 'All ordered word steps', 'Training CV', 'Gate failed; far below structure'],
                        ['GRU retrieval loss', 'Full-candidate cross entropy', 'Training CV', 'Predictions spread; no better retrieval'],
                        ['Word pretraining', 'Exported-word targets first', 'Training CV', 'No gain over fresh training'],
                        ['N-way analysis', 'Smaller random candidate pools', 'Replayed training CV', 'Real EEG near chance, below missingness']],
                 widths=[2286000, 3291840, 2286000, 2926080], size=1500,
                 callout='Training CV and reused validation are development scores, not untouched confirmation.',
                 notes='Later studies from the REPORT.md tables and each archive under docs/research_results. Every score here is a development score. Word-order used training CV selection and one evaluation on reused outer validation. Amplitude, GRU, loss-only and word-pretraining pilots did not evaluate outer validation. None used the test set.'),
            dict(title='Ordered-word ridge does not beat its controls',
                 subtitle='REUSED OUTER VALIDATION  /  ZuCo1 seed 42  /  512 queries  /  298 candidates  /  alpha chosen on training readers',
                 table=metric_rows(outer, 'word_order', [('Mean real EEG', 'mean_real'), ('Ordered real EEG', 'ordered_real'), ('Matched Gaussian', 'matched_gaussian'),
                                                         ('Word-order shuffled', 'word_order_shuffled'), ('Class shuffled', 'class_shuffled'),
                                                         ('Missingness only', 'missingness_only'), ('Structure only', 'structural_only')]),
                 widths=metric_widths, size=1600,
                 callout='Order adds one top-1 hit over the mean; Gaussian and order-shuffled score higher on R@1 and MRR.',
                 notes='Source: evidence_review_v1_20261101/outer_validation.csv. Training-fitted 16-component word PCA with mean or four position bins. Source word order is displayed order, not fixation chronology. Missingness and structure use annotation-derived information and are not EEG decoding. Chance Recall@1 is 1/298.'),
            dict(title='Relative amplitudes: both heads select zero EEG weight',
                 subtitle=cv_scope,
                 table=[['Condition', 'Embedding R@1', 'Embedding MRR', 'Known-class R@1', 'Known-class MRR']] +
                       [[label, percent(cv['amplitude/semantic_embedding', arm]['recall@1']), score(cv['amplitude/semantic_embedding', arm]['mrr']),
                         percent(cv['amplitude/known_sentence_class', arm]['recall@1']), score(cv['amplitude/known_sentence_class', arm]['mrr'])]
                        for label, arm in [('EEG only', 'eeg_only'), ('Gaussian only', 'gaussian_only'), ('Structure only', 'structural_only'),
                                           ('Structure + real EEG', 'structural_real'), ('Structure + Gaussian', 'structural_gaussian')]],
                 widths=[3017520, 1943100, 1943100, 1943100, 1943100], size=1600,
                 callout=f"Combined models equal structure alone. The {percent(cv['amplitude/known_sentence_class', 'structural_only']['recall@1'])} known-class score is structure, not EEG.",
                 notes='Source: evidence_review_v1_20261101/training_cv.csv and amplitude_audit_v1_20261101/training_summary.json. Settings were selected on the same readers, so scores are optimistic. Known-class is closed-set classification of repeated sentences. The reduced reader amplitude spread is an observation, not proof of subject alignment.'),
            dict(title='GRU: the loss change spreads predictions, not retrieval',
                 subtitle=cv_scope + '  /  epoch 8',
                 table=[['Condition', 'Cosine R@1', 'Cosine MRR', 'Cross-entropy R@1', 'Cross-entropy MRR']] +
                       [[label, percent(cv['gru', arm]['recall@1']), score(cv['gru', arm]['mrr']), percent(cv['gru_ce', arm]['recall@1']), score(cv['gru_ce', arm]['mrr'])]
                        for label, arm in [('Real EEG', 'real_eeg'), ('Mean EEG', 'mean_eeg'), ('Matched Gaussian', 'matched_gaussian'), ('Class shuffled', 'class_shuffled'),
                                           ('Word-order shuffled', 'word_order_shuffled'), ('Missingness only', 'missingness_only'), ('Structure only', 'structural_only')]],
                 widths=[2743200, 2011680, 2011680, 2011680, 2011680], size=1600,
                 callout='Distinct top-1 choices rise from 6.375 to 53 per reader; real beats Gaussian in only 3/8 readers.',
                 notes='Source: evidence_review_v1_20261101/training_cv.csv, gru_retrieval_loss_v1_20261101/prediction_concentration.json and baseline_prediction_audit/summary.json. Real beats Gaussian on Recall@1 in 3/8 readers under both objectives. Chance Recall@1 is 1/300. Both pilots failed their descriptive gate, which is a stopping rule, not a significance test.'),
            dict(title='Word pretraining adds no sentence-retrieval gain',
                 subtitle=cv_scope + '  /  sentence scores',
                 table=metric_rows(cv, 'word_pretrain', [('Pretrained real EEG', 'pretrained_real'), ('Fresh, same sentence updates', 'fresh_same_sentence'), ('Fresh, equal total updates', 'fresh_equal_steps'),
                                                         ('Pretrained Gaussian', 'pretrained_gaussian'), ('Pretrained class shuffled', 'pretrained_class_shuffled'),
                                                         ('Pretrained missingness', 'pretrained_missingness'), ('Structure only', 'structural_only')]),
                 widths=metric_widths, size=1600,
                 callout='Real EEG beats fresh same-sentence training in 0/8 readers and missingness in 0/8 readers.',
                 notes='Source: evidence_review_v1_20261101/training_cv.csv and participant_metrics.csv. Fixed stages of 128 word updates and 256 sentence updates; only the projection transfers. Equal update counts do not match FLOPs or sequence exposure. Counts are descriptive, not significance tests.'),
            dict(title='Word-task scores are a separate quantity',
                 subtitle='WORD CLASSES, NOT SENTENCES  /  Training-reader CV  /  Macro across 8 readers  /  Text is a target only',
                 table=[['Condition', 'R@1', 'R@5', 'R@10', 'MRR']] +
                       [[label, percent(word[arm]['recall@1']), percent(word[arm]['recall@5']), percent(word[arm]['recall@10']), score(word[arm]['mrr'])]
                        for label, arm in [('Pretrained real EEG', 'pretrained_real'), ('Pretrained Gaussian', 'pretrained_gaussian'),
                                           ('Pretrained missingness', 'pretrained_missingness'), ('Pretrained class shuffled', 'pretrained_class_shuffled')]],
                 widths=metric_widths, size=1800,
                 callout='Missingness alone scores higher overall; only 1/8 readers favor real EEG on word Recall@1.',
                 notes='Source: evidence_review_v1_20261101/word_task_macro.json and word_task.csv. These are word-class scores, not sentence accuracy. Frequent words and gaze-derived missingness can support them without demonstrated EEG language information. Reader folds repeat sentence classes, so no unseen-word claim follows.'),
            dict(title='Smaller candidate pools change difficulty, not the model',
                 subtitle='EXPECTED ACCURACY  /  Replayed training-reader CV  /  Uniform random distractors  /  Not a new cohort',
                 table=[['N', 'Real EEG', 'Gaussian', 'Class shuffled', 'Missingness', 'Structure', 'Chance']] +
                       [[n, percent(nway[n, 'pretrained_real']['expected_accuracy'], 2), percent(nway[n, 'pretrained_gaussian']['expected_accuracy'], 2),
                         percent(nway[n, 'pretrained_class_shuffled']['expected_accuracy'], 2), percent(nway[n, 'pretrained_missingness']['expected_accuracy'], 2),
                         percent(nway[n, 'structural_only']['expected_accuracy'], 2), percent(nway[n, 'pretrained_real']['chance'], 2)]
                        for n in (2, 5, 10, 20, 50, 100, 300)],
                 widths=[1188720, 1600200, 1600200, 1600200, 1600200, 1600200, 1600200], size=1600,
                 callout=f"Two-way real EEG is {percent(nway[2, 'pretrained_real']['expected_accuracy'], 2)}, below 50% chance. Structure's {percent(nway[2, 'structural_only']['expected_accuracy'], 2)} is annotation-derived, not EEG.",
                 notes='Source: nway_word_pretrain_v1_20261101/summary.csv. Exact expectations from frozen models, averaged within reader then across readers. Ties with the target count as misses. Compare each column with chance and with controls at the same N. Random distractors do not control sentence length.'),
            dict(title='Project brief: what the evidence covers',
                 subtitle='Coverage of the seven brief steps  /  Done means executed and reported, not that decoding succeeded',
                 table=[['Brief step', 'Evidence', 'Status'],
                        ['1. Data and splits', 'ZuCo 1.0 and 2.0 prepared; QC cohorts 3,349 and 5,884; reader-disjoint train, validation and test.', 'Done'],
                        ['2. Preprocessing', '105 ordered channels, 200 Hz, train-fit normalizers; upstream session processing is inherited.', 'Done'],
                        ['3. Representations', 'Frozen adapted LaBraM, ZuCo reconstruction SSL, supervised CNN, statistical baselines; no useful gain.', 'Done'],
                        ['4. Decoding and noise control', 'Fixed-pool sentence retrieval; Gaussian and shuffled controls; controlled negative result.', 'Done'],
                        ['5. Quality and robustness', 'ZuCo2 ridge only, one split and corruption seed; quality weighting lower in 8 of 9 conditions.', 'Partial'],
                        ['6. New subjects and datasets', 'Unseen readers tested; cross-release transfer exploratory; strict transfer not run.', 'Partial'],
                        ['7. Evaluation and analysis', 'Methods compared on clean data; degraded conditions cover ridge only; limitations documented.', 'Partial']],
                 widths=[2788920, 6858000, 1143000], size=1400,
                 callout='Open items: strict transfer, and degraded-condition coverage beyond ZuCo2 ridge.',
                 notes='Steps come from the project brief. Quality source: docs/research_results/retrieval.csv, where the only corrupted-input rows are ZuCo2 ridge runs: additive noise, channel dropout and temporal masking at three nonzero severities. Clean MRR: standard 0.023036, quality-aware 0.020230. Strict transfer needs participant-identity evidence.'),
            dict(title='Current conclusion: a controlled negative result',
                 subtitle='All scopes considered  /  Scores from different scopes are not ranked against each other',
                 cards=[('Supported', ['Reproducible pipeline with matched controls and verified source pairing.',
                                       'Strong retrieval from annotation-derived counts and durations, separate from EEG.']),
                        ('Not supported', ['A useful, consistent EEG gain beyond the strongest controls.',
                                           'Verified participant-independent transfer between ZuCo releases.',
                                           'One validated final accuracy.']),
                        ('Needed', ['A revised, falsifiable hypothesis with a frozen plan.',
                                    'Independent readers and stimuli for generalization claims.',
                                    'Untouched evaluation, not more tuning on this development history.'])],
                 callout='This does not show that EEG lacks language information; these configurations did not establish it.',
                 notes='Consistent with REPORT.md and evidence_review_v1_20261101/SUPERVISOR_NOTE.md. One seed, short budgets and minimal architectures cannot rule out every representation. Experiments remain stopped until a new task is scoped explicitly.')]
    return deck


def render_slide(number, spec):
    if 'table' in spec:
        content = [table(20, spec['table'], spec['widths'], spec['size'])]
    else:
        gap = 274320
        card = (WIDTH-2*gap)//3
        content = []
        for index, (heading, items) in enumerate(spec['cards']):
            x = LEFT+index*(card+gap)
            content.append(rectangle(10+2*index, x, TABLE_TOP, card, TABLE_BOTTOM-TABLE_TOP, BAND))
            paragraphs = [run(heading, 2400, TEAL, True)]+[run(item, 1900, INK) for item in items]
            content.append(text_box(11+2*index, x+137160, TABLE_TOP+137160, card-274320, TABLE_BOTTOM-TABLE_TOP-274320, paragraphs, after=1400))
    return slide_xml(number, spec['title'], spec['subtitle'], content, spec['callout'])


def build(output):
    if output.exists():
        raise FileExistsError(output)
    assert hashlib.md5(SOURCE_DECK.read_bytes()).hexdigest() == SOURCE_DECK_MD5
    found = claims()
    assert found == EXPECTED_CLAIMS, found
    specs = slides(load_evidence())
    assert len(specs) == 9
    with ZipFile(SOURCE_DECK) as source:
        names = source.namelist()
        parts = {name: source.read(name) for name in names}
    types = parts['[Content_Types].xml'].decode('utf-8')
    presentation = parts['ppt/presentation.xml'].decode('utf-8')
    relations = parts['ppt/_rels/presentation.xml.rels'].decode('utf-8')
    app = parts['docProps/app.xml'].decode('utf-8')
    next_id = max(int(i) for i in re.findall(r'<p:sldId [^>]*\bid="(\d+)"', presentation))+1
    next_rel = max(int(i) for i in re.findall(r'Id="rId(\d+)"', relations))+1
    new_types, new_ids, new_relations, new_parts = '', '', '', {}
    relation_type = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    for offset, spec in enumerate(specs):
        number = 27+offset
        new_parts[f'ppt/slides/slide{number}.xml'] = render_slide(number, spec)
        new_parts[f'ppt/slides/_rels/slide{number}.xml.rels'] = (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{relation_type}/slideLayout" Target="../slideLayouts/slideLayout7.xml" />'
            f'<Relationship Id="rId2" Type="{relation_type}/notesSlide" Target="../notesSlides/notesSlide{number}.xml" /></Relationships>').encode('utf-8')
        new_parts[f'ppt/notesSlides/notesSlide{number}.xml'] = notes_xml(spec['notes'])
        new_parts[f'ppt/notesSlides/_rels/notesSlide{number}.xml.rels'] = (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{relation_type}/notesMaster" Target="../notesMasters/notesMaster1.xml" />'
            f'<Relationship Id="rId2" Type="{relation_type}/slide" Target="../slides/slide{number}.xml" /></Relationships>').encode('utf-8')
        new_types += (f'<Override PartName="/ppt/slides/slide{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml" />'
                      f'<Override PartName="/ppt/notesSlides/notesSlide{number}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml" />')
        new_ids += f'<p:sldId r:id="rId{next_rel+offset}" id="{next_id+offset}" />'
        new_relations += f'<Relationship Id="rId{next_rel+offset}" Type="{relation_type}/slide" Target="slides/slide{number}.xml" />'
    parts['[Content_Types].xml'] = types.replace('</Types>', new_types+'</Types>').encode('utf-8')
    parts['ppt/presentation.xml'] = presentation.replace('</p:sldIdLst>', new_ids+'</p:sldIdLst>').encode('utf-8')
    parts['ppt/_rels/presentation.xml.rels'] = relations.replace('</Relationships>', new_relations+'</Relationships>').encode('utf-8')
    app, slide_count = re.subn(r'(<[^<>]*Slides>)26(</)', r'\g<1>35\2', app)
    app, notes_count = re.subn(r'(<[^<>]*Notes>)26(</)', r'\g<1>35\2', app)
    assert slide_count == 1 and notes_count == 1
    parts['docProps/app.xml'] = app.encode('utf-8')
    for data in new_parts.values():
        E.fromstring(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', ZIP_DEFLATED) as target:
        for name in names:
            target.writestr(name, parts[name])
        for name, data in new_parts.items():
            target.writestr(name, data)
    print(output)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output, pdf, manifest):
    if manifest.exists():
        raise FileExistsError(manifest)
    assert hashlib.md5(SOURCE_DECK.read_bytes()).hexdigest() == SOURCE_DECK_MD5
    assert hashlib.md5(SOURCE_PDF.read_bytes()).hexdigest() == SOURCE_PDF_MD5
    found = claims()
    assert found == EXPECTED_CLAIMS, found
    with ZipFile(SOURCE_DECK) as old, ZipFile(output) as new:
        old_names, new_names = set(old.namelist()), set(new.namelist())
        assert old_names <= new_names and new.testzip() is None
        changed = sorted(name for name in old_names if old.read(name) != new.read(name))
        assert changed == ['[Content_Types].xml', 'docProps/app.xml', 'ppt/_rels/presentation.xml.rels', 'ppt/presentation.xml'], changed
        added = sorted(new_names-old_names)
        assert len(added) == 36
        for name in new_names:
            if name.endswith(('.xml', '.rels')):
                E.fromstring(new.read(name))
        order = re.findall(r'<p:sldId [^>]*r:id="(rId\d+)"', new.read('ppt/presentation.xml').decode('utf-8'))
        assert len(order) == 35 and len(set(order)) == 35
        assert all(f'ppt/slides/slide{i}.xml' in new_names for i in range(1, 36))
        slide_text = {i: re.findall(r'<a:t>([^<]*)</a:t>', new.read(f'ppt/slides/slide{i}.xml').decode('utf-8')) for i in range(27, 36)}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report = dict(created='2026-11-05', scope='Saved evidence only; no training, tuning, inference or evaluation ran.',
                  builder=dict(path='scripts/consolidate_later_evidence.py', sha256=sha(Path(__file__))),
                  source_deck=dict(path=SOURCE_DECK.relative_to(ROOT).as_posix(), md5=SOURCE_DECK_MD5, pptx_sha256=sha(SOURCE_DECK), pdf_md5=SOURCE_PDF_MD5, pdf_sha256=sha(SOURCE_PDF)),
                  output_deck=dict(path=output.resolve().relative_to(ROOT).as_posix(), sha256=sha(output), slides=35),
                  output_pdf=dict(path=pdf.resolve().relative_to(ROOT).as_posix(), sha256=sha(pdf)),
                  source_parts_changed=changed, parts_added=added, slides_1_to_26_byte_identical=True,
                  stated_numbers_checked=found, evidence_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in SOURCES},
                  new_slide_titles={str(i): text[0] for i, text in slide_text.items()})
    manifest.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--manifest', type=Path)
    arguments = parser.parse_args()
    if arguments.manifest:
        verify(arguments.output, arguments.pdf, arguments.manifest)
    else:
        build(arguments.output)
