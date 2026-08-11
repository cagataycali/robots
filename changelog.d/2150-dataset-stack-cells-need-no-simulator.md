### Quality: run every dataset-stack unavailability cell without an optional backend

The nine cells that pin what `start_recording` reports when the LeRobot dataset
stack is unusable matter most on a partially provisioned install - and the
install most likely to be missing the lerobot extra is also the one most likely
to be missing a simulator. Their MuJoCo engine was a real `Simulation` behind
`pytest.importorskip("mujoco")`, on the premise that those cells need a compiled
model. They do not: the block runs before the lock and before any MuJoCo call,
so the pre-flight reads only a world sentinel, one robot and the two rollout
maps `_active_rollout_rates` prunes.

With the `mujoco` package blocked at the import system, the module went from
26 passed / 11 skipped to 39 passed / 0 skipped - the three MuJoCo cells were
among the 11, so on such an install its own mutation evidence did not hold
either. Replaces that factory with a `__new__` skeleton matching the Newton and
Isaac ones beside it, and adds a guard so the requirement cannot creep back.
Tests only - no library behaviour changes.
