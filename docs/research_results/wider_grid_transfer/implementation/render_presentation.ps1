param([Parameter(Mandatory=$true)][string]$InputDeck,
      [Parameter(Mandatory=$true)][string]$OutputDir)
$ErrorActionPreference = 'Stop'
$deckPath = (Resolve-Path -LiteralPath $InputDeck).Path
$outPath = [IO.Path]::GetFullPath((Join-Path $PWD $OutputDir))
if (Test-Path -LiteralPath $outPath) { throw "Use a fresh render directory: $outPath" }
New-Item -ItemType Directory -Path $outPath | Out-Null
$powerpoint = New-Object -ComObject PowerPoint.Application
$presentation = $null
try {
    $presentation = $powerpoint.Presentations.Open($deckPath, -1, 0, 0)
    $presentation.Export($outPath, 'PNG', 1280, 720)
    $presentation.SaveAs((Join-Path $outPath 'preview.pdf'), 32)
    Write-Output "Rendered $($presentation.Slides.Count) slides to $outPath"
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    $powerpoint.Quit()
}
