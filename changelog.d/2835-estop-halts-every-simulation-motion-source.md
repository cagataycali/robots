### Fixed: an emergency stop on a simulation halts teleoperation as well as policies

`SimulationDeviceDriver.onEmergencyStop` wrote `robot.policy_running = False`
over every robot in the world and stopped there. A `Simulation` mixes in
`TeleopMixin`, so a leader arm can be driving it from a thread that flag says
nothing about: the teleop loop polls `get_action()` and applies the result
through `send_action` on its own thread. On a teleoperated simulation the leader
was polled 26 more times after the emergency stop returned, with the teleop
session still marked running and the handler reporting the halt.

The simulation's own `cleanup` already stops teleoperation under the same guard
this handler now uses, because it is a motion source that cannot be left
running. Both sibling handlers hold an emergency stop to every source they own:
`reachy_mini_driver.onEmergencyStop` attempts torque-off and stop-motion and
logs a failure at CRITICAL, and `robot_driver.onEmergencyStop` reads the stop
verdict rather than discarding it.

Policies are stopped first, since that is a flag write which cannot block, so
the unblockable kill lands even when the bounded teleop join outlasts its
budget. Both stops are attempted even if one raises, and a source that did not
stop is logged at CRITICAL naming it and quoting the reason the stop gave. The
verdict is read through `teleop_mixin._stop_reported_stopped` rather than from
the envelope's status, because `_teleop_stats` derives that status from the
session counters and a session whose frames errored answers `"error"` after a
perfectly clean join. The receipt moves from `print()` to a log record: a safety
receipt written to stdout carries no level to alert on.
