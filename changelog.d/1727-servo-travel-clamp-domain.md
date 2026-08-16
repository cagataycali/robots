### Fixed: refuse a `max_relative_target` the driver cannot honor

`Robot(...)` forwarded the `max_relative_target` servo travel clamp to lerobot's
robot config without validating it, and the driver honors a narrower domain than
the config dataclass accepts. `nan` or `inf` disabled the clamp with no signal, a
negative limit inverted it into a fixed-magnitude step that ignores the requested
goal, `0` discarded every commanded motion while the rollout reported success, and
an `int` limit - type-correct against the field's `float` annotation under PEP 484's
numeric tower - raised a bare `TypeError(10)` at the first servo command. The limit
is now validated and normalized when the config is built, before the serial port is
opened, and the same domain applies to every value of a per-motor mapping. `None`
still means the clamp is disabled.
