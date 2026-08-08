# ZeroActionMonitor: a non-finite action stream is not a near-zero one

`capture.py` runs unchanged in two trees (a `git worktree` at `upstream/main`,
and the branch), each printing the tree it resolved `strands_robots` from and
dumping `facts-*.json`. `compose.py` builds the figure and asserts every
rendered number against those two dumps before saving:

- streams reported as the wrong fault: `3 of 5` -> `0 of 5`
- the honored MuJoCo rollout: identical emitted action dict, identical final
  joint positions to 12 dp, `44.126527 deg` of travel, render `max|delta| = 1/255`
  over 164 of 471,200 px

Every figure claim is measured; nothing is hand-typed.
