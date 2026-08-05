# Artifact: the huggingface_hub floor for the `hf` bucket CLI

`wheel_bisect.py` — bisects the published wheels for `huggingface_hub/cli/buckets.py`
and its registration in `cli/hf.py` (first present in 1.5.0).

`real_cli_runtime.sh` — runs the genuine `hf` binary from a shadow venv of each
release: 1.0.0 / 1.4.1 answer `hf buckets create` with
`Error: No such command 'buckets'` (rc=2); 1.5.0 answers rc=0.

`measure_floor.py` — drives `sync_dataset_to_bucket` against those real CLIs in a
given source tree, dumping JSON. Run once per tree (upstream/main and the branch).

`compose_figure.py` — composes `bucket_cli_floor.png` from the two dumps and
asserts every claim it renders before saving.
