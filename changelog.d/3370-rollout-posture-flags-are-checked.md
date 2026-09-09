### Fixed: the rollout's three posture flags are checked, not read by truthiness

`run_policy` tables the domains of its quantities - the horizon, the rate, the
seed, the substeps - and read all three of its *postures* raw. A posture selects
a branch rather than scaling a quantity, so there is nothing to clamp and no
partial effect, and every non-empty string is truthy. Measured on a MuJoCo
rollout, every row returning `status="success"`:

- `fast_mode="false"` (also `"no"`, `"off"`, `"0"`) ran 30 steps at 30Hz in
  0.006s where `fast_mode=False` takes 1.001s - the whole rollout issued as one
  burst instead of paced on a deadline, which falsifies both claims the deadline
  pacer exists to hold (`duration` in wall-clock seconds, `fast_mode=False` as
  real-time pacing).
- `reset_between="no"` reset between episodes, while `0` / `""` / `None` / `[]`
  did not - so every episode after the first began in the state the previous one
  ended in and was still recorded as an independent episode.
- `wbc_install_torque_control=None` withheld the torque shim a position-servo
  humanoid needs for a stable gait, which is the one thing that flag installs.

The two flags that default to `True` are the sharper pair: for them a falsy
non-boolean *removes* behaviour the rollout was going to get. The same
`fast_mode` field was already held to this domain by the mesh wire schema before
being forwarded into the same rollout, so a value an untrusted `tell()` could not
get past the transport was reachable from every local caller.

All three are now checked on the shared `boolean_flag_error` domain, in the shape
each surface answers in: a structured error from `run_policy`, `start_policy` and
the `run_policy` tool - which checks before its own recording step can replace a
dataset at `dataset_root` - and a `ValueError` from `PolicyRunner.run`, which is
drivable directly. `reset_between` is checked only when `n_episodes > 1`, the
condition under which it is read at all. Both declared postures of every flag are
unchanged.
