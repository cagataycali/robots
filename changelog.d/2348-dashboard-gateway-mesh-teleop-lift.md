### Fixed: a coordinator process can join the mesh without owning a robot

Fixes found while driving a fleet dashboard against the mesh. `robot_mesh`
now brings up a robot-less *gateway mesh* so coordinator processes
(dashboards, schedulers, loggers) can reach the fleet - previously every
action failed with `no local mesh found` despite live peers, and the
discovery wait now happens once at gateway bring-up rather than per call so
a call burst cannot stretch past the rate-limit window.
`start_teleop_receive` / `get_teleop_status` moved off the hardware `Robot`
into the shared `TeleopMixin` so a sim digital twin can follow a real leader
arm (`robot_name` scopes the target arm). Child sim-robot peers delegate
`execute`/`start` to the parent `Simulation` via a `_sim_parent` dataclass
field (a bare `SimRobot` has no `run_policy`, so addressable child peers
answered `unknown action: execute`), and the child `Mesh._read_state()`
extracts per-robot joints via the `_world` backref.

A gateway peer is no longer counted as a peer that failed to stop.
`Mesh.start` subscribes every peer to `strands/broadcast`, so the fleet
`emergency_stop()` fanout reaches a robot-less gateway too; with no robot to
halt it fell through to `{"ok": False, "error": "peer exposes no stop_task"}`
and the operator's own dashboard was named in the CRITICAL "robots may still
be executing" warning on every e-stop. A robot-less peer now answers
`{"ok": True, "stopped": [], "note": "no robot registered on this peer"}`. A
peer whose registered robot exposes no stop verb still reports failure.

### Fixed: `STRANDS_MESH_STREAM_HZ=0` no longer fails robot construction

The per-step telemetry throttle divided the environment value directly, so
`STRANDS_MESH_STREAM_HZ=0` raised `ZeroDivisionError` and a non-numeric value
raised `ValueError` - from `HardwareRobot.__init__`, failing every
`Robot(..., mode="real")` construction, and from the simulation `run_policy`
hook setup. Both now resolve the rate through the shared mesh domain check:
non-positive turns step publishing off (the spelling `STRANDS_MESH_CAMERA_HZ`
already uses), and an unusable value is logged with the variable and the
offending value named, then leaves publishing off.
