### Changed: a policy's optional hooks are read once, and a wrong type is refused by name

`Robot.run` and `Simulation.run` read `hz`, `realtime`, `gains`, `attach`, `detach`, `requirements` and
`needs_observation` once, before step one, and refuse a hook of the wrong type with a `TypeError` naming it.
Before, `hz = 0` ran at 30 Hz, `realtime = "no"` paced in wall time and `needs_observation = False` rendered every
step, all silently; `requirements() -> dict` failed deep in the check.

`Robot.stop(timeout=)` and `Task.stop(timeout=)` raise `TimeoutError` while the run is still going, instead of
returning `None` as if nothing ran. A run whose policy raised raises that error from `stop()` once; the next
`stop()` returns `None`.
