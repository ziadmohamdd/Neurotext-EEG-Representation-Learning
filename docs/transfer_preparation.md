# Prepare strict transfer while identity evidence is pending

Preparation is allowed before evidence arrives; strict training is not authorized
by preparation alone. This workflow leaves all identities and source signals
unchanged. It builds metadata/text targets and records the existing strict gate.
It does not perform exploratory transfer or bypass participant checks.

Run from the existing Windows checkout:

```powershell
$zuco1 = Get-Content data/processed/zuco1_batch/summary.json -Raw | ConvertFrom-Json
$zuco2 = Get-Content data/processed/zuco2_batch/summary.json -Raw | ConvertFrom-Json
$transferDir = "data/processed/transfer_plan_$(Get-Date -Format yyyyMMdd_HHmmss)"

python -m eeg_decoding.data.transfer_plan --zuco1 data/manifests/zuco1_duration_qc.jsonl --zuco2 data/manifests/zuco2_duration_qc.jsonl --text-zuco1 $zuco1.text_cache --text-zuco2 $zuco2.text_cache --output $transferDir
if ($LASTEXITCODE -ne 0) { throw "Transfer preparation failed" }

Get-Content "$transferDir/plan.json" -Raw
```

Expected with the current cohorts: 9,233 references to existing signals, a shared
sentence target cache, both directions listed as `not_executed`, and overall
`blocked_pending_participant_identity`. This status is expected, not a software
failure. Unique sentence/candidate counts come from actual manifests, not an
assumed count in the paper. No EEG arrays or neural checkpoints are copied.

Checks include signal validity, documented microvolt units, channel order,
sampling frequency, comparable task conditions and dataset versions. Frozen text
caches must agree on model/revision metadata and dimension; shared sentence
vectors must agree within float32 tolerance. If they disagree, do not silently
substitute one cache: regenerate one shared frozen text cache under a single
documented revision. Stored cohort digests bind source data and metadata.

The plan also reports target observations/texts absent from the entire source
cohort. These are planning counts, not an executed unseen-content transfer
condition. The current direct CLI transfer uses the inclusive target pool and
must disclose shared texts. Library support for source-unseen target filtering
must be explicitly selected and logged in a separate experiment if used.

`configs/transfer_ridge_only.yaml` specifies six initial jobs: both transfer
directions with real, Gaussian and shuffled ridge conditions at seed 42. It
requests no extra robustness suite. Its input recipe is
`configs/labram_full_sentence.yaml` (full recordings). Do not confuse the
19-channel LaBraM adaptation with these 105-channel statistical inputs.

When a maintainer response/public source arrives, review its cohort scope and
record the evidence. Assign genuine cross-release person IDs, exclude any
overlapping people for strict transfer and regenerate the prepared plan. Do not
add version-specific fake identities merely to satisfy the gate. Model selection
and EEG normalization must use source training/validation data only.

No command in the preparation workflow launches the planned jobs. The training
code still checks strict identity compatibility. Reusing text targets is not
target-EEG fitting, and preparation is not a research result.
