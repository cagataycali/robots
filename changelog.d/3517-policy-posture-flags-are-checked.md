### Fixed: a policy posture flag is checked, not coerced with `bool()`

`pad_short_actions` (`lerobot_local`, `lerobot_async`) and `walk` (`wbc`) each
select one of two postures rather than scaling a quantity, and each constructor
stored the caller's value through `bool(...)`. Every non-empty string is truthy,
so `bool("false")` is `True`: `pad_short_actions="false"` selected the PADDING
posture, which commands every actuator the model produced no value for to `0.0`
- an absolute target on a LeRobot `<motor>.pos` follower or a MuJoCo position
actuator, so those joints travel there. Measured on a 6-actuator SO-101 driven
by a 4-value chunk, `"false"` swung joints 5 and 6 by -1.06 rad and -1.02 rad
where `False` held both at 0.0000 rad. `walk="false"` likewise loaded and
preferred the locomotion policy instead of running the balance policy alone,
and `None` / `0` took the other branch without ever being a declared spelling
of it. Nothing raised and nothing logged on either half, so the wrong posture
was indistinguishable from the right one until the robot moved.

All three are now checked with `boolean_flag_error` - the domain the package
already applies to the `mesh.iot` provisioning flags and, through
`SimEngine._validate_posture_flags`, to `fast_mode` - and the value is stored as
given rather than converted. `lerobot_async` also gained the `Args:` entry for
`pad_short_actions` it was missing.
