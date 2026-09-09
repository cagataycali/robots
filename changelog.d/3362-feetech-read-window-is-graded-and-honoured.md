### Fixed: a Feetech read window that no read can wait for is refused, and the one a caller sets reaches the bus

`FeetechBus` graded `baud_rate` and not `timeout`, though the two are opened on
the same `serial.Serial` call and the rationale in the class docstring is the
same for both. pyserial takes `0`, `nan`, `inf` and `None` verbatim as a
timeout, and each of them makes `_read_one` see an empty buffer that its retry
loop cannot tell from a servo that never answered, so `sync_read` omits the
motor and logs "no verified reply": on a fake port that honours its timeout, a
six-servo arm reported 0 of 6 joints under `timeout=0` and 6 of 6 under the
default. The two pyserial does refuse - a negative and a string - it refuses
from inside `connect`, naming neither the bus nor the parameter. `timeout` is
now held to `positive_finite_number_error` beside `baud_rate`, and the default
is the named `DEFAULT_TIMEOUT_S`.

`FeetechDriver` popped no `timeout` at all, so a caller who lengthened the
window for a slow servo had it recorded in `self._extras` while the bus opened
at the default - the case the driver's own `motor_ids` comment names, a keyword
that changes nothing while the caller believes the arm is configured. It is now
forwarded to the bus and graded naming the driver, as `baud_rate` already was.
