### Fixed: G1 `send_action` refuses a joint command the joint cannot honor

`G1Driver.send_action` (and the `g1_send_action` agent tool that passes the
model's dict to it verbatim) copied the caller's per-joint `q`, `kp`, `kd`, `dq`
and `tau` onto the `LowCmd_` frame after checking only that each was a finite
number; the docstring deferred magnitude to an arm-SDK client that does not
exist (F-004, CWE-1284). `{"left_knee": {"q": 30.0, "kp": 500.0}}` was a valid,
CRC'd frame. Every field is now held to a per-joint envelope taken from the
vendor's `g1_29dof.urdf` (travel, peak torque, peak speed) plus gain ceilings of
twice / five times the stiffest reference `kp` / `kd`; a value outside it is
refused with a reason naming the joint, field, value and bounds, the frame is
never built, and nothing is published. Refused, not clamped: a clamped target
still moves a standing humanoid somewhere the caller did not ask for.
