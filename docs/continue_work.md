# Continue experiments without waiting for correspondence

The user chose to continue without waiting for the maintainer reply. No reply is
required for the diagnostics below or the explicitly exploratory transfer path.
Existing results are preserved. No cross-release identities are invented.

## First: diagnose training capacity

From the activated Windows project terminal:

```powershell
git pull origin main
if ($LASTEXITCODE -ne 0) { throw "Git pull failed" }
& .\scripts\run_diagnostics.ps1
```

This runs up to 200 full-batch gradient steps on eight distinct sentence
observations from one **training** subject in each version. The ordinary
seed-42 subject split determines eligible subjects; validation/test observations
are not used. Cropping follows `configs/local.yaml` (8 seconds). The same CNN,
projection and contrastive loss are used, with float32, no weight decay and no
gradient accumulation to test optimization capacity. This is a deliberately
easy memorization problem, not a replica of every original training recipe.

Outputs go to fresh `results/diagnostics/...` folders: selected signal references,
signal/text audit, progress history, diagnostic checkpoint and final JSON. They
do not enter the research evaluation CSV. Original EEG is not copied or changed.
The audit lists actual text and sentence IDs, duration, flat-channel counts,
maximum magnitude, duplicate signal pairs and target similarity. These checks
cannot establish original sentence-clock timing or annotation correctness.

Send the final per-dataset summary. `MemorizationAchieved=True` means 100% top-1
on these same eight training examples, not generalization. It shows this small
mapping can be learned but does not prove all training code or source labels are
correct. A false result is a reason to inspect loss, learning rate, normalization,
architecture and inputs before more large runs; it is not by itself proof of a
bug. Gaussian/noise can also be memorized, so success is not semantic evidence.

## Next: exploratory transfer without verified participant independence

The existing shared plan/cache can be used unchanged. Its recorded
`blocked_pending_participant_identity` status refers to **strict** transfer;
the separately named exploratory protocol records the unresolved identity scope.

The six-job configuration covers both directions with real, Gaussian and shuffled
ridge controls. Run after the diagnostic review:

```powershell
$planPath = "data/processed/transfer_plan_20261029_213051/plan.json"
$plan = Get-Content $planPath -Raw | ConvertFrom-Json
$transferRuns = "results/exploratory_transfer_$(Get-Date -Format yyyyMMdd_HHmmss)"

python -m eeg_decoding.cli run-experiments --manifest $plan.manifest --text-cache $plan.text_cache --config configs/labram_full_sentence.yaml --experiments configs/transfer_exploratory_ridge.yaml --output $transferRuns --execute
if ($LASTEXITCODE -ne 0) { throw "Exploratory transfer failed" }

Write-Host "Results: $transferRuns/retrieval.csv"
```

For a single manual run, `train-baseline` and `train-alignment` accept
`--protocol transfer_exploratory --source zuco1 --target zuco2` (or reverse).
Normalization, SSL/projection/ridge fitting and model selection use source
training/validation only. Both versions must still share channel order,
sampling rate and task; known participant overlap is still rejected.

All new evaluations retain `protocol: transfer_exploratory`, report
`identity_audited: false`, and disclose shared sentence counts and the inability
to establish unseen-person transfer. CSVs retain the distinct protocol label;
the full assessment is in evaluation JSON and model metadata. The inclusive
target pool contains shared source texts, so these runs must not be described
as unseen-sentence transfer either. Do not mix these results with strict transfer
or claim that participant independence has been verified.

Strict `--protocol transfer` keeps its original identity requirement. If primary
evidence becomes available later, it can support a separate strict experiment;
it is no longer a prerequisite for continuing the chosen exploratory work.
