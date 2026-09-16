### Fixed: a config field the class derives for itself is no longer demanded of the caller

`Robot(name, mode="real")` resolves lerobot's config dataclass for the robot
type and refuses up front when a required field no forwarded kwarg supplied --
naming the parameter the caller states rather than letting the dataclass report
a lerobot internal. "Required" was read as "has no default", which is not the
same question: `dataclasses.field(init=False)` names a value the class computes
in `__post_init__`, so it carries no default *and* is absent from `__init__`.

lerobot 0.6.2 declares one such field, `UnitreeG1Config.sim_env`, so
`Robot("g1", mode="real", robot_ip=...)` was refused with `missing required
parameter(s) ['sim_env']` for a config that constructs fine -- and the remedy
that refusal printed, `Robot(..., sim_env=...)`, raises `TypeError: __init__()
got an unexpected keyword argument 'sim_env'`: a dead end whichever way the
caller turned. The comment above the scan states the intended invariant --
"this changes which sentence a refused call gets, not which calls are refused"
-- which a field the constructor never accepted had quietly broken.

The rule now has one owner, `_requires_a_caller_value`, which asks `field.init`
alongside the two defaults. The camera-option scan reads it through the same
owner, where a derived option was both advertised as accepted and refused for
its absence, so the two scans cannot come to disagree about what a caller can
be asked for. `strands_robots.training.lerobot` already reads `field.init` for
the same question about training-config kwargs.
