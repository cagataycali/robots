### Fixed: a scene mutation is refused during a blocking `run_policy`, not just a `start_policy`

`_require_no_running_policy` is the safety gate every MuJoCo scene mutation runs
first, and its own docstring states the hazard: an XML round-trip swaps
`world._model` / `world._data` while a live rollout holds pointers into the old
arrays, so the rollout segfaults on its next `mj_step`.

It derived its population from `self._policy_threads`, and that table records
only the rollouts `start_policy` submits - the blocking `run_policy` drives on
its caller's thread and registers no future. So `add_robot`, `remove_object`,
`move_object`, `add_camera`, `remove_camera`, `set_gravity`, `set_timestep`,
`reset` and the rest were refused during a Future-backed rollout and *accepted*
during a blocking one, on a scene that rollout was stepping - while
`list_policies_running`, the mesh `status` command and its state topic all
reported the rollout in flight. Measured on `so100` at 20 Hz, all eight
global-scope mutations flipped from refused to accepted purely on the launch
shape.

That is the two-sources drift #2833 closed for the reporting surfaces, reached
through the gate instead of the report: `_active_policy_robots` owns the union of
the Future table and the per-robot `policy_running` claim, and
`_rollouts_in_flight` delegates there rather than re-deriving it. The gate now
reads that same population, so a rollout that is reported as running is a
rollout whose scene is protected.

The rollout's own driving thread is exempt, on the reasoning that already makes
`self._lock` an `RLock`: a rollout mutates the scene itself between episodes
(`PolicyRunner` calls `sim.reset()` at the top of each one), and the driver
racing itself is not the hazard. The claim is recorded by `_drive_rollout`, the
body the blocking and Future-backed entry points share, so it names the thread
that actually steps the physics in either shape.
