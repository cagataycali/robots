### Fixed: a Reachy Mini action naming a key no axis answers is refused, not moved without it

`ReachyDriver.send_action` builds the daemon's head pose from six keys read with
a zero default, so a key the driver has no actuator for was dropped before any
check saw it. The gate above it - refusing an action that names *none* of the
nine axes - exists so an action that moved nothing is not reported as a success,
and that outcome was reachable straight through the gate, since one known key
satisfies it.

For the head it is worse than a no-op. The daemon's head command is a whole
pose, not a delta, so a dropped head axis is not left alone but commanded to
zero:

    send_action({"head_pitch": 20.0, "head_rol": 15.0})
    -> head_pose for pitch 20, roll 0        status="success"

The axis names most likely to be spelled that way are this package's own: the
`reachy_look` tool takes `pitch`/`roll`/`yaw`, `reachy_antennas` takes
`left`/`right` and `reachy_body_turn` takes `yaw`, while the driver's action keys
spell the same axes `head_pitch`, `antenna_left` and `body_yaw`. So
`{"head_pitch": 20.0, "roll": 15.0}` levelled the head it was asked to tilt,
`{"head_yaw": 20.0, "left": 30.0}` never moved an antenna, and
`{"body_yaw": 10.0, "yaw": 30.0}` left the head where it was - each under a
success envelope naming the groups it did send.

A key outside the accepted set is now refused by name, the shape
`earthrover.send_action` already uses for its drive channels, `ur`'s
`targets_from_action` for its joints, `franka`'s for its joint keys and
`crazyflie`'s `action_to_setpoint` for its flight components. It is checked after
the all-unknown gate rather than before it, so each refusal diagnoses one fault:
an action naming no axis at all is told what to send, and an action that mostly
parsed is told which key was dropped. The reason names the substitution as well
as the key, because "ignored" and "commanded to zero" are different outcomes for
whoever plans the next gesture.
