# Build the final NeuroText deck (v3) with local Microsoft PowerPoint. Reads only saved figures; runs no experiment.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_final_deck.ps1 -Out docs/presentation/NeuroText_final_v3_20261117.pptx
param([Parameter(Mandatory = $true)][string]$Out, [string]$PreviewDir = '')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$out = [IO.Path]::GetFullPath((Join-Path $root $Out))
$pdf = [IO.Path]::ChangeExtension($out, '.pdf')
if ((Test-Path $out) -or (Test-Path $pdf)) { throw "Refusing to overwrite $out or $pdf" }
$figs = Join-Path $root 'docs\paper_figures'

function C([string]$hex) { $r = [Convert]::ToInt32($hex.Substring(0, 2), 16); $g = [Convert]::ToInt32($hex.Substring(2, 2), 16); $b = [Convert]::ToInt32($hex.Substring(4, 2), 16); return $r + 256 * $g + 65536 * $b }
$NAVY = C '1F4E79'; $INK = C '1D2433'; $MUTED = C '5B6475'; $ORANGE = C 'D9822B'; $SOFT = C 'F2F5F9'; $LINE = C 'D8DEE8'; $WHITE = C 'FFFFFF'; $TEAL = C '2A9D8F'; $ICE = C 'CADCFC'
$FONT = 'Segoe UI'; $SEMI = 'Segoe UI Semibold'
$W = 960; $H = 540; $M = 48

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Add(0)
$pres.PageSetup.SlideWidth = $W; $pres.PageSetup.SlideHeight = $H
$script:n = 0

function New-Slide([bool]$dark = $false) {
    $script:n++
    $s = $pres.Slides.Add($script:n, 12)
    $s.FollowMasterBackground = 0
    $s.Background.Fill.Solid()
    $s.Background.Fill.ForeColor.RGB = $(if ($dark) { $NAVY } else { $WHITE })
    return $s
}

function T($s, $x, $y, $w, $h, [string]$text, [double]$size = 16, $color = $INK, [bool]$bold = $false, $align = 1, [string]$font = $FONT) {
    $b = $s.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $f = $b.TextFrame; $f.WordWrap = -1; $f.AutoSize = 0
    $f.MarginLeft = 0; $f.MarginRight = 0; $f.MarginTop = 0; $f.MarginBottom = 0
    $r = $f.TextRange; $r.Text = $text
    $r.Font.Name = $font; $r.Font.Size = [single]$size; $r.Font.Color.RGB = $color; $r.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $r.ParagraphFormat.Alignment = $align
    return $b
}

function Bullets($s, $x, $y, $w, $h, [string[]]$items, [double]$size = 15, $color = $INK) {
    $b = T $s $x $y $w $h ($items -join "`r") $size $color
    $r = $b.TextFrame.TextRange
    $r.ParagraphFormat.Bullet.Visible = -1; $r.ParagraphFormat.Bullet.Character = 8226
    $r.ParagraphFormat.Bullet.Font.Color.RGB = $ORANGE
    $r.ParagraphFormat.SpaceAfter = 8
    $r.IndentLevel = 1
    $b.TextFrame.Ruler.Levels(1).FirstMargin = 0; $b.TextFrame.Ruler.Levels(1).LeftMargin = 16
    return $b
}

function Box($s, $x, $y, $w, $h, $fill, $shape = 5) {
    $b = $s.Shapes.AddShape($shape, $x, $y, $w, $h)
    $b.Fill.ForeColor.RGB = $fill; $b.Line.Visible = 0
    if ($shape -eq 5) { $b.Adjustments.Item(1) = 0.08 }
    return $b
}

function Badge($s, $x, $y, [string]$label, $fill = $ORANGE, $d = 30) {
    $c = $s.Shapes.AddShape(9, $x, $y, $d, $d)
    $c.Fill.ForeColor.RGB = $fill; $c.Line.Visible = 0
    $r = $c.TextFrame.TextRange; $r.Text = $label; $r.Font.Name = $SEMI; $r.Font.Size = 13; $r.Font.Color.RGB = $WHITE
    $c.TextFrame.MarginLeft = 0; $c.TextFrame.MarginRight = 0
    return $c
}

function Title($s, [string]$kicker, [string]$title) {
    if ($kicker) { $null = T $s $M 30 ($W - 2 * $M) 18 $kicker.ToUpper() 11 $ORANGE $true 1 $SEMI }
    $null = T $s $M 50 ($W - 2 * $M) 40 $title 26 $NAVY $false 1 $SEMI
    $null = T $s $M ($H - 26) 500 14 'NeuroText  ·  What ZuCo EEG-to-text scores measure' 9 $MUTED
    $null = T $s ($W - $M - 60) ($H - 26) 60 14 "$($script:n)" 9 $MUTED $false 3
}

function Pic($s, [string]$name, $x, $y, $w) {
    $p = $s.Shapes.AddPicture((Join-Path $figs $name), 0, -1, $x, $y)
    $p.LockAspectRatio = -1; $p.Width = $w
    return $p
}

function Notes($s, [string]$text) { $s.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = $text }

function Stat($s, $x, $y, $w, $h, [string]$value, [string]$label, $accent = $NAVY) {
    $null = Box $s $x $y $w $h $SOFT
    $null = T $s ($x + 18) ($y + 16) ($w - 36) 50 $value 36 $accent $false 1 $SEMI
    $null = T $s ($x + 18) ($y + 72) ($w - 36) ($h - 84) $label 12.5 $INK
}

# 1. Title
$s = New-Slide $true
$null = T $s $M 150 ($W - 2 * $M) 20 'NEUROTEXT  ·  EEG-TO-TEXT AUDIT' 12 $ORANGE $true 1 $SEMI
$null = T $s $M 180 ($W - 2 * $M) 110 'What do ZuCo EEG-to-text retrieval scores measure?' 38 $WHITE $false 1 $SEMI
$null = T $s $M 300 ($W - 2 * $M) 30 'An audit of task mixing, list position and controls' 18 $ICE
$null = T $s $M 440 ($W - 2 * $M) 20 'Ziad Mohamed  ·  November 2026' 13 $ICE
Notes $s 'This project started as an attempt to decode sentences from reading EEG. It became an audit of how such decoding is scored, because the scores turned out to come mostly from shortcuts.'

# 2. Answer first
$s = New-Slide; Title $s 'The answer in one slide' 'Most of the score comes from shortcuts, not from brain content'
$cw = ($W - 2 * $M - 2 * 20) / 3
Stat $s $M 120 $cw 190 '19.5%' "10-way score from knowing only the corpus, with no EEG. GLIM's published outputs score 21.5%. Chance is 10%."
Stat $s ($M + $cw + 20) 120 $cw 190 '51.1%' "10-way score from copying list neighbours' text on a random split, with no EEG."
Stat $s ($M + 2 * ($cw + 20)) 120 $cw 190 '≤ 2.4 pts' 'The largest gain EEG can still add once eye tracking is known, under fair block holdouts.' $ORANGE
$null = Box $s $M 340 ($W - 2 * $M) 70 $NAVY
$null = T $s ($M + 22) 352 ($W - 2 * $M - 44) 50 'Confirmed once, under a frozen plan, on 85 sentences never used before: all three pre-stated claims held.' 16 $WHITE $false 1 $SEMI
Notes $s 'Three numbers carry the talk. First, a no-EEG predictor that only knows which corpus a sentence comes from nearly matches a published model. Second, random splits let list neighbours give away the answer. Third, under a fair split, EEG adds at most about 2.4 points beyond gaze. A final one-time test on untouched sentences confirmed it.'

# 3. Background
$s = New-Slide; Title $s 'Background' 'Decoding which sentence someone is reading'
$null = Bullets $s $M 120 470 330 @(
    'ZuCo 1.0 and 2.0: EEG and eye tracking recorded while 30 people read sentences (12 + 18 readers).',
    'A model maps EEG to a sentence embedding (MiniLM) and must pick the right sentence out of N candidates: N-way retrieval.',
    'Chance is 1/N, so 10% at 10-way.',
    'Published models report scores above chance. We asked what those scores actually measure.')
$qs = @(@('1', 'Task mixing', 'How much of a score comes from telling corpora or tasks apart?'),
    @('2', 'Split design', 'Do random sentence splits let list position stand in for content?'),
    @('3', 'Controls', 'Under a fair split, does EEG beat matched noise and eye tracking?'))
for ($i = 0; $i -lt 3; $i++) {
    $y = 120 + $i * 112
    $null = Box $s 560 $y 352 98 $SOFT
    $null = Badge $s 578 ($y + 16) $qs[$i][0]
    $null = T $s 622 ($y + 16) 270 22 $qs[$i][1] 15 $NAVY $false 1 $SEMI
    $null = T $s 622 ($y + 42) 270 50 $qs[$i][2] 12.5 $INK
}
Notes $s 'N-way retrieval: the model sees the EEG for one sentence and ranks N candidate sentences. We report expected accuracy over all ways of drawing the distractors, with ties counted as misses.'

# 4. Finding 1
$s = New-Slide; Title $s 'Finding 1  ·  Task mixing' 'Mixed-task scores mostly tell corpora apart'
$null = Pic $s 'fig1_task_mixing.png' $M 115 560
$null = Bullets $s 630 120 282 330 @(
    "GLIM's public outputs: 21.46% at 10-way; the published figure is 21.91%.",
    'Knowing only "movie review or biography", no EEG: 19.53%.',
    'Inside any single task GLIM scores 9.73–14.25%.',
    'Reviews vs biographies: 87–94% at 2-way. Biography tasks vs each other: 52–61%.',
    'Two more published models (BART, T5) score at chance and noise level.') 13.5
Notes $s 'We scored GLIM''s own published test outputs without reading any EEG. The corpus-only line is an exact calculation: a predictor that ranks every sentence from the query''s corpus first and guesses inside it. It matches GLIM at 2-way and 4-way and comes within 2 points at 10-way.'

# 5. Finding 2
$s = New-Slide; Title $s 'Finding 2  ·  Split design' 'Random splits leak list position'
$null = Pic $s 'fig2_position.png' $M 115 520
$null = Bullets $s 600 120 312 330 @(
    'Every reader saw the sentences in the same fixed order, and EEG drifts across a session.',
    "Copying a held-out sentence's list neighbours, no EEG: 51.09% at 10-way.",
    'EEG predicts list position: r = 0.80 (ZuCo 1.0) and 0.88 (ZuCo 2.0).',
    'The same CNN: 23.10% on a random split, 10.02% under block holdout (noise 9.72%).') 13.5
Notes $s 'On a random split, a held-out sentence''s neighbours in the reading list are usually in training. Because EEG drifts slowly, it carries position information, and position can be turned into a retrieval score without any sentence content.'

# 6. Method
$s = New-Slide; Title $s 'Method' 'Hold out whole blocks, and compare with strong controls'
$cells = @(); for ($i = 0; $i -lt 30; $i++) { $cells += $(if ($i -lt 8 -or $i -ge 22) { 'T' } elseif ($i -lt 11 -or $i -ge 19) { 'B' } else { 'X' }) }
for ($i = 0; $i -lt 30; $i++) {
    $fill = switch ($cells[$i]) { 'T' { $NAVY } 'B' { $LINE } 'X' { $ORANGE } }
    $null = Box $s ($M + $i * 28.8) 118 24 24 $fill 1
}
$null = T $s $M 148 230 16 'Training sentences' 11 $NAVY $true
$null = T $s ($M + 8 * 28.8) 148 90 16 'Buffer' 11 $MUTED $true
$null = T $s ($M + 11 * 28.8) 148 230 16 'Held-out block, scored 10-way inside it' 11 $ORANGE $true
$null = T $s $M 172 ($W - 2 * $M) 20 '13 blocks, each held out once: 255 sentences. Training leaves out the block and 10 list positions on each side.' 12.5 $INK
$rows = @(@('Control', 'What it rules out'), @('Matched noise', 'Channel amplitudes and slow drift: same per-recording strength, no content'),
    @('Shuffled pairs', 'Pipeline artefacts; breaks the EEG–sentence pairing'), @('List position, no EEG', 'Place in the list, even inside a block'),
    @('Timing and word count, no EEG', 'Sentence length and reading time'), @('Gaze and word properties, no EEG', 'Word length, frequency, fixations and reading time'),
    @('Partial check', 'Removes position, word count and reading time from predictions'))
$tb = $s.Shapes.AddTable($rows.Count, 2, $M, 205, $W - 2 * $M, 260)
$tb.Table.ApplyStyle('{2D5ABB26-0587-4C30-8999-92F81FD0307C}')
$tb.Table.Columns(1).Width = 270; $tb.Table.Columns(2).Width = $W - 2 * $M - 270
for ($i = 1; $i -le $rows.Count; $i++) {
    for ($j = 1; $j -le 2; $j++) {
        $cell = $tb.Table.Cell($i, $j)
        $cell.Shape.Fill.Visible = -1
        $cell.Shape.Fill.ForeColor.RGB = $(if ($i -eq 1) { $NAVY } elseif ($i % 2 -eq 0) { $SOFT } else { $WHITE })
        $r = $cell.Shape.TextFrame.TextRange; $r.Text = $rows[$i - 1][$j - 1]
        $r.Font.Name = $(if ($i -eq 1 -or $j -eq 1) { $SEMI } else { $FONT }); $r.Font.Size = 12.5
        $r.Font.Color.RGB = $(if ($i -eq 1) { $WHITE } else { $INK })
        $cell.Shape.TextFrame.MarginTop = 5; $cell.Shape.TextFrame.MarginBottom = 5
    }
}
Notes $s 'Each test had a written plan frozen before it ran, a leakage audit that refused to run on any failure, and a stop rule. Matched noise keeps each recording''s amplitude, so it sits above chance: it is the bar for content beyond amplitude and drift.'

# 7. Finding 3
$s = New-Slide; Title $s 'Finding 3  ·  Controls' 'Under block holdout, EEG does not beat matched noise'
$null = Pic $s 'fig3_block_tests.png' $M 110 520
$null = Bullets $s 600 120 312 330 @(
    'Whole-sentence EEG 11.25% vs noise 12.13%.',
    'Word-locked EEG 14.55% vs noise 16.25%.',
    'First-fixation EEG 12.81% vs noise 9.72%: not clear (interval −1.5 to 7.8).',
    'Eye tracking and word properties with no EEG: 26.05%, the strongest arm by far.') 13.5
Notes $s 'Three different EEG representations, each with its own frozen plan. None was clearly above matched noise. Gaze with no EEG beat all of them, which says that what is decodable here is reading behaviour, not EEG content.'

# 8. Effect limits
$s = New-Slide; Title $s 'How big could an effect still be?' 'A null result is only as useful as its interval'
$cw = ($W - 2 * $M - 2 * 20) / 3
Stat $s $M 108 $cw 122 '≤ 1.2 pts' 'Whole-sentence EEG over matched noise'
Stat $s ($M + $cw + 20) 108 $cw 122 '≤ 2.8 pts' 'Word-locked EEG over matched noise'
Stat $s ($M + 2 * ($cw + 20)) 108 $cw 122 '≤ 2.4 pts' 'First-fixation EEG added on top of eye tracking' $ORANGE
$null = Pic $s 'fig4_readers.png' $M 250 520
$null = T $s 610 270 302 160 'Upper ends of 95% intervals at 10-way. Averaging more readers did not lift whole-sentence or word-locked EEG above noise. The first-fixation gap that grew with readers (right) reversed on the reserved sentences.' 13 $INK
Notes $s 'These are the largest gains the data still allow. Any EEG contribution that gaze does not already carry is small under this design; a study powered to find it would need many more sentences.'

# 9. Confirmation
$s = New-Slide; Title $s 'Confirmation  ·  scored once' 'All three claims held on 85 never-used sentences'
$null = Pic $s 'fig5_confirmation.png' $M 112 560
$claims = @(@('C1', 'EEG does not beat matched noise', 'No pipeline clear'), @('C2', 'Gaze beats word-locked EEG', '+12.1 points (3.96 to 20.38)'),
    @('C3', 'EEG adds little to gaze', 'Upper end 1.2 < preset 3 points'))
for ($i = 0; $i -lt 3; $i++) {
    $y = 118 + $i * 76
    $null = Box $s 630 $y 282 66 $SOFT
    $null = Badge $s 642 ($y + 18) $claims[$i][0] $TEAL 30
    $null = T $s 682 ($y + 10) 220 20 $claims[$i][1] 12.5 $NAVY $false 1 $SEMI
    $null = T $s 682 ($y + 32) 220 30 "Confirmed  ·  $($claims[$i][2])" 11 $INK
}
$null = T $s 630 352 282 110 'The one development lead, first fixation, reversed: 12.90% vs 16.79% for noise. Before scoring, the plan stated how each claim would be confirmed or contradicted.' 12 $MUTED
Notes $s 'These four reserved blocks were the last clean held-out data in ZuCo normal reading. The pipelines were reused unchanged, the plan was frozen with a hash, and the blocks were scored once. Fifteen of the 100 reserved sentences belonged to an older test set that stays closed, so 85 were scored.'

# 10. Rigour
$s = New-Slide; Title $s 'Rigour' 'How the results were kept honest'
$cards = @(@('Frozen plans', 'Each test was written down and its config hashed before running, with a fixed stop rule.'),
    @('Leakage audit', 'The run refuses to start if any test sentence or buffer enters training.'),
    @('Protected files', '11,462 earlier results, caches and EEG sources hashed before and after the final run: unchanged.'),
    @('Tests and reruns', '430 automated tests: planted signals are found, position-only signals are rejected. Earlier results reproduce exactly.'))
$cw = ($W - 2 * $M - 20) / 2
for ($i = 0; $i -lt 4; $i++) {
    $x = $M + ($i % 2) * ($cw + 20); $y = 120 + [math]::Floor($i / 2) * 160
    $null = Box $s $x $y $cw 140 $SOFT
    $null = Badge $s ($x + 20) ($y + 20) "$($i + 1)" $NAVY
    $null = T $s ($x + 64) ($y + 22) ($cw - 84) 24 $cards[$i][0] 16 $NAVY $false 1 $SEMI
    $null = T $s ($x + 64) ($y + 52) ($cw - 84) 80 $cards[$i][1] 13 $INK
}
Notes $s 'The code, plans, hashes and results are in the public GitHub repository. No raw EEG or ZuCo text is redistributed.'

# 11. Reporting controls
$s = New-Slide; Title $s 'Contribution' 'Five reporting controls for EEG-to-text retrieval'
$ctl = @('Report N-way scores within each task, beside a corpus-only or task-only ceiling.',
    'Hold out whole list blocks with buffers, not random sentences.',
    'Use matched noise that keeps per-recording amplitude as the bar, not only chance.',
    'Report a gaze and word-property baseline with no EEG, and the gain of EEG over it.',
    'Report scores against the number of readers averaged, and the largest effect the intervals allow.')
for ($i = 0; $i -lt 5; $i++) {
    $y = 118 + $i * 70
    $null = Badge $s $M ($y + 4) "$($i + 1)" $ORANGE 34
    $null = T $s ($M + 52) ($y + 10) ($W - 2 * $M - 52) 40 $ctl[$i] 16 $INK
}
Notes $s 'These controls are cheap to run and would let readers tell content decoding from shortcuts.'

# 12. Limits and next steps
$s = New-Slide; Title $s 'Scope' 'Limitations and next steps'
$cw = ($W - 2 * $M - 30) / 2
$null = T $s $M 118 $cw 22 'Limitations' 17 $NAVY $false 1 $SEMI
$null = Bullets $s $M 150 $cw 300 @('One dataset, ZuCo; the final test used new sentences but the same readers.',
    '85 reserved sentences give wide intervals.', 'Linear models on summary features.',
    'This does not show that ZuCo EEG lacks language information, only that these scores do not show it.') 14
$null = T $s ($M + $cw + 30) 118 $cw 22 'Next steps' 17 $ORANGE $false 1 $SEMI
$null = Bullets $s ($M + $cw + 30) 150 $cw 300 @('Test a new EEG reading dataset with block holdouts and these controls from the start.',
    'Report every score beside its shortcut ceilings.', 'Try nonlinear models under the same controls.') 14
Notes $s 'No clean held-out data remains in ZuCo normal reading, so any further claim needs new data.'

# 13. Conclusion
$s = New-Slide $true
$null = T $s $M 120 ($W - 2 * $M) 20 'CONCLUSION' 12 $ORANGE $true 1 $SEMI
$null = T $s $M 150 ($W - 2 * $M) 160 'ZuCo EEG-to-text retrieval scores mostly measure corpus, task and list position. Under fair controls, EEG adds at most a few points beyond eye tracking.' 28 $WHITE $false 1 $SEMI
$null = T $s $M 380 ($W - 2 * $M) 20 'Code, frozen plans and results: github.com/ziadmohamdd/Neurotext-EEG-Representation-Learning' 13 $ICE
Notes $s 'Thank you. Questions welcome.'

$pres.SaveAs($out, 24)
$pres.SaveAs($pdf, 32)
if ($PreviewDir) {
    $dir = [IO.Path]::GetFullPath($PreviewDir); New-Item -ItemType Directory -Force $dir | Out-Null
    for ($i = 1; $i -le $pres.Slides.Count; $i++) { $pres.Slides($i).Export((Join-Path $dir ("slide{0:D2}.png" -f $i)), 'PNG', 1280, 720) }
}
$count = $pres.Slides.Count
$pres.Close(); $app.Quit()
"Wrote $count slides to $out and $pdf"
