param([int]$Steps = 200)
$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$runTag = Get-Date -Format "yyyyMMdd_HHmmss"
$diagnosticRows = @()

foreach ($dataset in @("zuco1", "zuco2")) {
    $batchSummary = Get-Content "data/processed/${dataset}_batch/summary.json" -Raw | ConvertFrom-Json
    $diagnosticDir = "results/diagnostics/${dataset}_tiny_${runTag}"
    Write-Host "Training-subset diagnostic: $dataset (not held-out evaluation)"
    python -m eeg_decoding.diagnostics --manifest "data/manifests/${dataset}_duration_qc.jsonl" --text-cache $batchSummary.text_cache --config configs/local.yaml --sentences 8 --steps $Steps --output $diagnosticDir
    if ($LASTEXITCODE -ne 0) { throw "Diagnostic failed: $dataset" }
    $diagnosticReport = Get-Content "$diagnosticDir/diagnostic.json" -Raw | ConvertFrom-Json
    $diagnosticRows += [PSCustomObject]@{
        Dataset = $dataset
        Folder = $diagnosticDir
        MemorizationAchieved = $diagnosticReport.memorization_achieved
        InitialLoss = $diagnosticReport.initial.loss
        FinalLoss = $diagnosticReport.final.loss
        TrainingRecallAt1 = $diagnosticReport.final.training_metrics.'recall@1'
        TrainingMRR = $diagnosticReport.final.training_metrics.mrr
    }
}
$diagnosticRows | Format-List
