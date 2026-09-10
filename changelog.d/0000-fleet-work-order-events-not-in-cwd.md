### Fixed

- `examples/fleet/05_work_order_dispatch.py` writes its default `work_order_events.jsonl` under the temp dir instead of the current directory (`--events` still picks any path).
