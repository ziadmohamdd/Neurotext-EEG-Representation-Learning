# Terminal display and verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/show_accuracy.py --study word-order
```

This only reads the saved validation summary. It does not fit, tune or evaluate test data.

Before fitting, 11 focused tests passed (`tests/test_word_order.py` and `tests/test_spectral_pilot.py`). After the run, the verification script reproduced all 3,584 saved ranks from seven frozen models, checked all 168 inner-training scores and all eight reader-fold partitions, and confirmed Gaussian validation masks match. SHA256 checks confirmed all 1,109 protected prior files remained unchanged. No test features were extracted and no test evaluation occurred.

See [verification evidence](validation_checks.json), [full pilot report](README.md) and [archive checksums](archive_manifest.json). This count concerns focused software checks, not a new full-suite run or evidence of scientific performance.

The frozen descriptive improvement screen failed. No further tuning or test evaluation follows this pilot. The main report and presentation are preserved without silently replacing their earlier evidence.
