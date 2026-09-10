### Fixed: the G1 driver reads `rt/lowstate.motor_state`, so its 29 PD-controlled joints are observed

`G1Driver._on_lowstate` decoded `rt/lowstate` into the IMU and the hardware
layout id and never touched `motor_state`, the array the robot's 29 joints
report `q` / `dq` / `tau_est` / `temperature` through. The driver therefore
published PD targets on `rt/lowcmd` under gains up to `_SDK_KP` while observing
no joint at all: `send_action` could not check a target against the measured
pose, and `_ControlLoop._call_policy` handed a policy an observation of
`{mode_machine, fsm_id, battery, imu}`, so no proprioceptive policy - which is
every locomotion and whole-body controller - could run on the humanoid.

The twin `Go2Driver._on_lowstate` already read the same array the same way, per
`GO2_JOINT_INDEX`, and published it on its `sensors` verb, so one SDK family
answered a joint read on the quadruped and not on the humanoid.

The G1 now reads it over `_G1_JOINT_INDEX` and exposes `joints` on the
`sensors` verb and in the control loop's policy observation. The decode itself
moves to `decode_motor_state` in `drivers/base.py` and both drivers call it, so
the Go2's private copy is deleted rather than duplicated - the same
one-owner shape the shared telemetry coercers took. The shared decoder keeps
the family's rule that a default is not a reading: a field the frame does not
carry reads `None` rather than `0.0`, which on a joint would be a plausible
pose, and a `motor_state` that is absent, bytes-like, or answers no slot reads
`None` rather than an empty record.
