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
