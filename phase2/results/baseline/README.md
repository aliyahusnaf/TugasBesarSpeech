# Owner 3 — actual inference evidence

All runs use the pinned multilingual Whisper small checkpoint on CPU INT8.
Complete raw predictions are local and ignored by Git because masked-name utterances can reveal identities.
Raw prediction hashes below bind those local files without publishing them. No test inference or WER/CER scoring was performed.

| Run ID | Completed validation rows | Elapsed seconds |
| --- | ---: | ---: |
| whisper_small_baseline_001 | 235 | 503.47 |
| whisper_small_beam5_reload_001 | 3 | 12.44 |
| whisper_small_beam5_smoke_001 | 3 | 12.28 |
| whisper_small_reload_001 | 3 | 11.17 |
| whisper_small_rms_trim_reload_001 | 3 | 11.28 |
| whisper_small_rms_trim_smoke_001 | 3 | 11.48 |
| whisper_small_smoke_001 | 3 | 8.95 |

`whisper_small_baseline_001` covers every validation row with the original reference text.
The six three-row runs are execution/reload checks, not full experimental comparisons.
Some smoke runs occurred alongside the baseline; elapsed times are operational logs, not controlled speed benchmarks.
Two initial comparison smoke runs recorded a dirty worktree while documentation was being edited; the adapter code was unchanged. All fresh-process reloads recorded a clean worktree.

- `reload_check.json`: exact prediction/config/model-hash matches for all three configurations on the same three validation IDs.
- `file_api_check.json`: whole-file API output matched its segment baseline.
- `whisper_small_baseline_001/examples.csv`: five reviewed, non-identifying examples from S03_T02_0001 through S03_T02_0005. This small excerpt is not an accuracy estimate.
- `prediction_inventory.json`: hashes of complete local prediction files and published metadata.

Full suite: 37/38 tests passed. The existing source-fingerprint test fails on LF/CRLF differences; run metadata records the in-memory newline-equivalence check. All 11 new tests passed.
Owner 2 should review waveform/processor compatibility. Owners 4/5 should run full comparisons and score the same validation IDs before selecting a final configuration. Human cross-review is still pending.
