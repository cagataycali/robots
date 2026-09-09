### Fixed: a rollout can be stopped on every simulation backend, through one verb

`stop_policy` is the counterpart every `start_policy` docstring names, and it
existed on the MuJoCo engine alone. On Newton, Isaac and any backend written
against the documented `SimEngine` ABC the attribute was simply absent, so the
two remote stop paths worked around it instead of reading its answer:
`Mesh._dispatch` probes `hasattr(r, "stop_policy")` and answered `"peer exposes
no stop_task"` for a simulation it could have stopped, while the Device Connect
`stop` RPC re-derived the `was_running` verdict inline from the per-robot flag -
a second construction of the answer `SimRobot.request_policy_stop` exists to keep
in one place ("EVERY stop path goes through here ... so they cannot drift to
different answers about whether a rollout was halted").

That inline fallback also read the robot registry as `sim._world.robots`, which
is the MuJoCo and Newton spelling. The Isaac engine keeps its robots in
`_robots` and its `_world` is the Isaac `World`, which has no `.robots` at all,
so the read raised `AttributeError` and the handler's recovery path reported it
as `"The simulation changed under the stop loop"` - a race that had not happened
- before a single robot was asked to stop.

`SimEngine.stop_policy` now owns the question on every backend, the way
`run_multi_policy` has since #2157: validate the name, then delegate the flag
write to `_request_policy_stop`, the third member of the
`_make_run_policy_hook` / `_release_run_policy_hook` seam that already raises and
lowers the same flag around a rollout the shared facade drives. Newton overrides
it, so its stop is a real one that reports `was_running` and increments the
durable `policy_stops` counter. Isaac inherits the base default, which refuses
and names the class: its per-robot record carries a bare `policy_running` flag
and not the durable claim, and a bare flag write is exactly what a worker before
its first frame overwrites (#2833), so it states why it cannot help rather than
reporting a stop it cannot keep. The refusal names remedies this backend does
have - `run_policy(n_steps=...)` and `run_policy(stop_when=...)`. The Device
Connect `stop` RPC now reads that verb and enumerates through the ABC's own
`list_robots`, so no path reaches into a backend's private registry.

The counterpart's own description is corrected in the same change, because it is
what sent callers looking for a stop that was not there. `start_policy`'s summary
line read "Start policy execution in a background thread (non-blocking)" while
the next line read "Default implementation: synchronous passthrough to
`run_policy`" - and the summary line is what `help()`, an IDE tooltip and the
rendered reference show. `docs/api-reference.md` called it an async rollout
unconditionally and listed `stop_policy` as a `SimEngine` action, and
`docs/troubleshooting.md` prescribed it as the cure for a hanging agent.
Measured on the base default with `duration=3.0`, the call returns after 4.6s
reporting `Policy complete`, against 0.001s and `Policy started (async)` on
MuJoCo - so the prescribed remedy for a hang was the hang. All three surfaces now
name the condition, and `describe()["methods"]["start_policy"]` states which of
the two implementations the engine in hand has, so a caller can tell at runtime
instead of reading the source.
