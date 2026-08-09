### Fixed: the dataset-publication posture flags are checked instead of read by truthiness

`sync_dataset_to_bucket` allowlist-validates `bucket` and `run_id` before it runs
anything, and its own docstring says so. The three flags in that same signature -
`create`, `private` and `delete` - reached `hf buckets create` / `hf sync` raw, and so
did `private` on `DatasetRecorder.push_to_hub` beside it. Each selects a posture on a
remote store, so each now goes through `boolean_flag_error`, the domain the mesh
provisioning entry points already apply to their own capability flags.

Read by truthiness they failed toward the permissive posture in both directions.
`delete="false"`, `"no"`, `"off"` and `"0"` - the spellings an operator reaches for
when opting out - are truthy, so each appended `--delete` and mirror-deleted remote
files absent locally; `private=0`, `""`, `None` and `[]` are falsy, so each dropped
`--private` and created the bucket public. Every one returned `status="success"`.
The refusal is placed ahead of the CLI probe, so a flag outside its domain now reports
identically whether or not `hf` is installed and reaches neither subprocess.

`sync_to_bucket` forwards all three flags and so inherits the refusal; a structural
test keeps it that way. Both booleans still build a byte-identical command line.
