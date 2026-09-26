### Fixed: an agent demo raises the run the command gate paused, rather than reporting it as done

`cmd_vel` is a gated command surface, so an agent driving one with no operator
to ask has its run PAUSED rather than finished: `result.stop_reason` is
`"interrupt"`, nothing has moved, and `result.interrupts` holds the question --
whose `reason` carries both the resume form and the
`STRANDS_ROS2_COMMAND_ALLOW=<surface>` line that stands in for an operator.
`examples/rosbridge/curiosity_agent.py` and `examples/ros2/deepracer_agent.py`
printed that list under `Agent completed: ...` and exited 0. Measured against a
live `rosbridge_server` driving a real robot, the rosbridge demo printed its
completion line with the robot 0.0 mm from where it started and not one
`cmd_vel` message on the wire; with the surface pre-approved the same script
drove it 3373.3 mm over 83 messages.

Both demos now raise on a paused run, carrying the question rather than a
verdict, and the rosbridge demo's header names the pre-approval its own gate
needs -- as its DeepRacer twin already did. The quickstart that runs it
(`docs/ros2/rosbridge-robot.md`) exports the same value and says what it buys.

`tests/test_examples_never_report_a_refused_command.py`, renamed from
`tests/test_examples_ros2_demos_raise_the_refused_command.py` for the behaviour
it grades rather than the transport it started on, now covers both caller
shapes: an error envelope returned to a programmatic caller, and a paused agent
run. Each demo is one row; no ROS 2, no DDS and no model call -- the bridge
class and `strands.Agent` are stubbed at the seams the scripts import them from.
