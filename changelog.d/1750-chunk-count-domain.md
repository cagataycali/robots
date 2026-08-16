### Fixed: a per-inference chunk count the consumer cannot execute is refused instead of floored

`actions_per_step` (how many actions of one inference chunk a consumer executes
before re-querying) and `actions_per_chunk` were stored verbatim by both LeRobot
providers and only reconciled on the read path, where `Policy.execution_horizon`
resolves them through `max(1, int(...))`. That floor turned a count no consumer
can execute into `1` under a successful call: `0`, `-5` and `False` all became a
single-action horizon, `2.7` was truncated to `2`, `"4"` was string-coerced, and
`None` / `nan` / `inf` / `[4]` raised a bare `TypeError` / `ValueError` /
`OverflowError` from a property read inside the rollout loop.

For `lerobot_local` the floor was worse than the default it replaced.
`_auto_detect_actions_per_step` treats any value other than the default `1` as a
horizon the caller pinned deliberately and returns without adopting
`config.n_action_steps`, so on a checkpoint trained to replay a 100-action chunk
open-loop, `actions_per_step=0` both skipped that adoption and floored the
horizon to `1` - re-querying the model every step, the out-of-distribution
operation the auto-detection exists to prevent - where the default would have
executed the trained 100. It also flipped `is_chunk_emitting()` to `False`,
silently disabling async-RTC latency masking for a chunk-emitting model.

Both counts are now validated where the caller supplies them, through a shared
`chunk_count_error` domain so the two providers cannot drift apart, and before
any checkpoint is fetched: in each constructor and in `lerobot_local.preflight`,
which the rollout entry points run first, so the same mistake surfaces as a
structured error rather than a raise. `actions_per_chunk` is checked too, since
it is the default for `actions_per_step`. `None` remains `lerobot_async`'s
documented "use `actions_per_chunk`". The floor in `resolve_chunk_length` is
unchanged: it is the default for a duck-typed chunk source that never passed
through a provider constructor.
