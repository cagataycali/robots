### Fixed: a Crazyflie action naming a key no setpoint carries is refused, not flown without it

`action_to_setpoint` reads every flight component with a zero default, so a key
the driver does not command was dropped before any check saw it and the
component it named rested at zero. The gate above it - refusing an action that
names *none* of `ACTION_KEYS` - exists because an all-zero setpoint "would latch
the aircraft into a hover the caller never asked for and report success for it",
and that outcome was reachable straight through the gate, since one known key
satisfies it:

    action_to_setpoint({"vx": 0.0, "height": 1.5})
    -> ("send_velocity_world_setpoint", (0.0, 0.0, 0.0, 0.0))

`height` is this module's own internal spelling of the hover key `z`, so the
request most likely to be spelled that way is the one that reaches the resting
setpoint. `{"vx": 0.5, "vyaw": 2.0}` translated without the yaw, and
`{"z": 1.0, "vX": 0.6}` hovered without the velocity - each reported as success.

A key outside `ACTION_KEYS` is now refused by name, the shape `earthrover`'s
`send_action` already uses for its drive channels and `ur`'s
`targets_from_action` for its joints. It is checked after the all-unknown gate
rather than before it, so each refusal diagnoses one fault: an action with no
known key is told what to send, and an action that mostly parsed is told which
component was dropped. Both refusals name the accepted keys and their units
from one constant, so the two cannot drift apart.
