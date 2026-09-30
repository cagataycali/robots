### Added: `ManipulationOptional`, a mixin for simulation backends without manipulation

A backend with no joints, objects or rendering - a sensor-only or remote
platform - still had to implement every manipulation-centric abstract member of
`SimEngine`, and `robot_joint_names -> list[str]` could not say "unsupported":
an empty list reads as a robot with no joints.

`strands_robots.simulation.capabilities.ManipulationOptional`, mixed in before
`SimEngine`, supplies `add_object`, `remove_object` and `render` as
`unsupported_by_backend` error results (built by the new `unsupported_result`)
and `robot_joint_names` as a `CapabilityNotSupported` raise (a
`NotImplementedError` carrying the capability and member). It declares only the
four core capabilities, keeps the shared parameter order of every member it
supplies, and refuses at class creation a `CAPABILITIES` claim that is still
backed by one of its refusals - including through `functools.wraps`, a partial,
or a subclass that reinstates the refusal - so a backend that claims a
capability must override the member that delivers it.
