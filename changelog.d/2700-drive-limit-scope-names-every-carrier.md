### Fixed: the drive-contract scope markers name every class that carries a limit

`docs/rosbridge-integration.md` and `RosbridgeRobot.drive` both marked the velocity clamp and
the `max_duration` ceiling as belonging to the rosbridge bridge alone, and said in prose that
`RosBridgedRobot.drive` and `RtpsRobot.drive` "carry none of the three". `AckermannRosRobot`
declares both, as `max_speed` and a `max_duration` of its own. The scope marker is the whole
point of those bullets - an unmarked guarantee reads as the fleet's, and a marked one tells a
reader which platform they have to supply the ceiling for themselves - so a reader who has just
been told the duration ceiling is one bridge's own carries the 30 s hold the page documents over
to a DeepRacer and is refused by that car's 10 s ceiling. The failure direction is the safe one,
but it is an unexplained refusal in the field, and the same reader plans around a velocity clamp
they have been told the car does not have. The fleet's own pages already stated both halves:
`docs/ros2-integration.md` documents the car's clamp and ceiling while the rosbridge page denied
they existed anywhere else.

`tests/mesh/test_drive_contract_fleet_scope.py` exists to grade that prose against a
measurement rather than against a hand-kept list, but its universe was a hardcoded list of the
three `Twist` bridges, so the fourth drive-owning class sat outside everything it checked and the
false claim passed green. The surveyed set is now derived from the package - a class that owns
`drive` and lives in a public module of `strands_robots.mesh` is a shipped platform bridge - and
the lookup is by attribute rather than by a `def drive` in the class body, so a bridge that
inherits `drive` from a shared base is still found while a private base module, which declares a
contract but is not a platform anyone drives, stays out. A fifth class fails the inventory until
it is given a case.

Three of the five guarantees are read without touching the message layout and are measured over
every drive-owning class: a refused input and a hold past a ceiling each forward no call at all,
and a velocity clamp makes two different over-ceiling requests forward the identical call, which
is as true of a servo pair as of a `Twist`. The clamp probe previously read
`calls[0]["fields"]["linear"]["x"]` and raises `KeyError: 'linear'` on a `ServoCtrlMsg`, so it
could not have been evaluated on that class even if the class had been listed. The two probes
that do read `Twist` content, the single-shot latch and the trailing zero, stay scoped to the
three `Twist` bridges for the reason `tests/mesh/test_bridge_stop_tool_parity.py` already gives:
an Ackermann car halts with a zero `ServoCtrlMsg`, not a zero `Twist`.

The measured split is now three-way - fleet-wide, carried by some, carried by this bridge alone -
because a limit two platforms declare is neither of the outer two and filing it under either
misinforms a reader of the other platforms. Both surfaces say which is which: the two bullets are
marked `(not on every mobile base)` and name `AckermannRosRobot`, and the docstring paragraph
opens `Not carried by every mobile base:` and credits the car's own lower ceiling. Two derived
rules hold them there - a bullet may claim `this bridge only` only when the survey finds no other
carrier, and every other carrier of a scoped guarantee has to be named. No library behaviour
changes; the only non-test edit is a docstring.
