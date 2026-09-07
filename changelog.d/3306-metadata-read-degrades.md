### Fixed: a dataset metadata file that cannot be decoded is reported, not raised

`json.load` fails in more ways than `json.JSONDecodeError`: bytes the declared
encoding does not describe raise `UnicodeDecodeError`, and a number longer than
`sys.get_int_max_str_digits` raises a plain `ValueError`. Four readers of a
dataset's `meta/info.json` / `meta/stats.json` named only the JSON one, so a
truncated or partially-synced file escaped instead of degrading:
`verify_dataset` lost its whole report - including every defect it had already
found - on exactly the corruption it exists to flag, `run_policy`'s
parquet-truth gate raised past the tool envelope, and both dataset probes
aborted `LerobotTrainer.validate`. All four now grade the read by `ValueError`,
matching the `meta/info.json` drift check that already did.
