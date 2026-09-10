### Fixed: a native `run_policy` runs the `Policy` its own seam types

`HardwareDriver.run_policy` types its first argument `Policy`, but no native
driver could actually run one. `Policy` declares `get_actions` and its
synchronous `get_actions_sync` wrapper, and no `step`/`__call__` -- so the G1 and
Go2 admission (`step` or a bare callable) turned every built policy away with
`policy_object must be callable or expose a .step() method`, and UR, which did
resolve `get_actions_sync`, then refused the answer one step later because
`get_actions` returns an action *chunk* and its loop demanded a dict. A rollout
on UR ended at step 0 having commanded nothing.

`policy_step()` in `strands_robots.drivers.base` is now the one owner of that
resolution: it accepts a built `Policy`, an object exposing `step`, or a bare
callable, and reduces a returned chunk to the action a step commands -- the same
first-action convention the lerobot path applies. G1, Go2 and UR all route their
admission *and* their per-step call through it, so the set a driver admits and
the set its loop can call cannot drift apart again. `instruction` now reaches
`get_actions_sync(observation, instruction)` on all three; G1 and Go2 previously
dropped it with `del instruction`. The untyped shapes are unchanged: a `step`
object or a bare callable still runs.
