### Fixed: an Ackermann car refuses a yaw it cannot execute instead of sending a halt

`AckermannRosRobot.drive` converts `(linear, angular)` through a bicycle model
that maps every `abs(linear)` below the rest threshold (`1e-3` m/s) onto the zero
servo pair. That conversion is right - a stock Ackermann platform steers its
front wheels and cannot yaw about its own axis, and `atan2` near `v = 0` would
amplify noise into hard steering - but the zero pair is exactly what `stop()`
publishes. A `drive(linear=0.0, angular=1.0)` therefore left on the wire as
`{"angle": 0.0, "throttle": 0.0}`, byte-identical to a halt, under
`status="success"`: a rotate-in-place request and a stop were indistinguishable
to the vehicle and to the caller, who went on to plan from a heading the car
never reached. The drive tool's own description already stated that the car
"cannot turn in place"; the call accepted it anyway.

`drive()` now refuses such a pair by name, quoting the threshold and what to send
instead, ordered with the other domain refusals and ahead of the `init_services`
handshake so an unexecutable request cannot be what switches the car into manual
mode. The threshold is read from the constant the conversion uses, so the door
and the conversion cannot come to disagree about what rest is. `drive(0, 0)`
still means rest, because that is what it asked for, and any yaw carrying a
linear speed still converts as before.
