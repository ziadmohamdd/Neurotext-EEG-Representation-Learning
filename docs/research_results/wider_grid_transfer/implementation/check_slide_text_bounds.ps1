param([string]$InputDeck='docs/presentation/NeuroText_consolidated.pptx')
$ErrorActionPreference='Stop'
$ppt=New-Object -ComObject PowerPoint.Application
$deck=$null
try {
    $deck=$ppt.Presentations.Open((Resolve-Path -LiteralPath $InputDeck).Path,-1,0,0)
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $range=$shape.TextFrame.TextRange
                if ($range.BoundWidth -gt $shape.Width+2 -or $range.BoundHeight -gt $shape.Height+2) {
                    [pscustomobject]@{Slide=$slide.SlideIndex;Shape=$shape.Name;
                        BoxWidth=$shape.Width;TextWidth=$range.BoundWidth;BoxHeight=$shape.Height;
                        TextHeight=$range.BoundHeight;Text=$range.Text}
                }
            }
        }
    }
} finally {
    if ($null -ne $deck) { $deck.Close() }
    $ppt.Quit()
}
