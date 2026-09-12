### Fixed: a Microduck action naming a key no intent carries is refused

`action_to_wire` built each robotd intent frame from `action.get(key, default)`,
and every frame carries its whole group, so an unknown key beside a known one was
not ignored -- it was replaced by the resting default and sent, reported as
`success`. `{"vx": 0.15, "yaw": 0.6}` (`yaw` is how the `get_status` pose block
spells the heading) sent `robot.move{vx:0.15,vy:0,vyaw:0}`: straight past the turn
the caller asked for. A 14-joint `MICRODUCK_JOINT_NAMES` action reached robotd as a
`robot.head` frame built from its four head axes with the other ten joints dropped
-- the per-joint stream `run_policy` and `start_task` refuse by name, arriving
through the intent path instead.

An action key this driver has no intent for is now refused, naming the dropped keys and
the accepted vocabulary, matching the same gate on the UR, Earthrover, Franka and
Crazyflie drivers. It is checked only when something else in the action did parse,
so each refusal diagnoses one fault: an action naming no intent at all still gets
`send_action`'s "nothing to send" message, and an action whose every key is real
reaches the wire unchanged.
