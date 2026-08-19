### Fixed: a CI apt step that hangs is bounded by the step, not by the job

`Test and Lint` reported `FAILURE` having never run a test, twice inside one
80-minute window: both jobs were reaped by the job's own `timeout-minutes: 45`
while still inside `Install system dependencies`, with lint and tests `skipped`
and nothing in the log naming apt. The step's retry loop is driven by
`apt-get update`'s exit status, so it rides out a mirror that answers wrongly
but not one that never answers - on a hang the `if` never evaluates, the next
attempt and the backoff are unreachable, and the diagnostic `echo` never
prints.

The looped `apt-get update` is now wrapped in `timeout`, which turns a hang
into the non-zero exit the loop already handles, and both apt steps
(`test-lint.yml`, `agent-api-check.yml`) declare their own `timeout-minutes` so
a stall is reaped on the step that caused it rather than aggregated into a
rollup `FAILURE` carrying no reason. Bounds are sized from the observed
distribution rather than from the command's nominal cost: `update` finishes in
~5s even on slow runs, while `install` fetching ffmpeg's ~115 packages reached
455s of a 45-minute budget, so `install` is bounded by the step and not by a
per-command timeout that would reap healthy runs. Pinned by
`tests/test_apt_steps_are_bounded.py`.
