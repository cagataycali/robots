### Fixed: `stop_recording` refuses a non-string `bucket` or `run_id` instead of raising

`stop_recording(bucket=123)` (or a list, dict, bool or float, and the same for
`run_id`) raised a bare `TypeError` from the bucket allowlist check, on an open
session and on the idle re-sync path alike. Both are now refused with the
standard `status="error"` result naming the parameter, before anything is
finalized, as `push_to_hub` and `private` already were. `None` still means
"not given".
