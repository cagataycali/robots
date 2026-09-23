### Fixed: the documented real-mode call grader reads a native driver's own keywords

`tests/test_docs_real_mode_invocations.py` graded every documented
`Robot(name, mode="real", ...)` against the robot's lerobot config dataclass,
which is the wrong roster for a call that builds a native driver
(`driver="strands"`, spelled or declared by the registry): the factory hands
those keywords to the driver as `**kwargs` and the driver reads the ones it
knows by `kwargs.pop()`. On `so101` that misread both ways - `calibration=`,
honoured by `FeetechDriver`, was reported as a defect, and `use_degrees=`, a
lerobot field the driver keeps as an unread extra, passed. The grader now
resolves the driver as the factory does and, for a native one, accepts the
factory's own parameters plus what the driver's `__init__` reads, derived from
its signature and its `kwargs` reads. The lerobot path is unchanged.
