# GH #2239 — an abandoned fixture work item decided whether the run ended

`capture.py` drives the real `Robot.start_task` path on two trees. It reads
whichever executor each tree's own fixture module imports, builds the same
`Robot` shape, wedges bring-up so the rollout never finishes, gives up on the
future after 2s, calls `shutdown(wait=False)`, and then lets the interpreter try
to exit under a 45s kill.

Measured (`facts.json`):

| | main | this PR |
|---|---|---|
| executor the fixture builds | `ThreadPoolExecutor` | `DaemonThreadExecutor` |
| the wait | `TimeoutError` after 2.0s | `TimeoutError` after 2.0s |
| non-daemon threads left | `['test_arm_executor_0']` | none |
| interpreter exited | **no — killed at 45s** | yes, exit 0 |
| wall clock | 45.06s | 2.78s |

The verdict is identical. Only the exit differs.

`mutate.py` is the mutation table: 7 plausible regressions x 2 test arms.
7 of 7 are caught by the new module; 6 of 7 are invisible to the 201
pre-existing cases in the five hardware modules involved.

`compose.py` builds the figure and asserts every drawn number against
`facts.json` before saving.
