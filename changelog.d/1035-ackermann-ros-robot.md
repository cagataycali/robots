### Added: `AckermannRosRobot` - Ackermann ROS 2 cars (AWS DeepRacer) as strands robots

`RosBridgedRobot` covers differential-drive bases (Twist on `cmd_vel` +
odometry); Ackermann cars expose neither. The new bridge keeps the same
`drive(linear, angular, duration)` agent contract and converts through a
bicycle model to normalized servo commands (`ServoCtrlMsg` on the DeepRacer),
runs a declarative `init_services` handshake once before the first command
(the DeepRacer's manual-mode two-step, preconfigured in `from_deepracer()`),
clamps speed, rejects over-long holds loudly, and always trails timed
commands with a zero servo message so a timed drive cannot leave the car
driving (a bare single-shot command latches until stop, matching raw servo
semantics). Conditional `get_scan` tool; no `get_pose` (the stock platform has
no odometry).

`drive` and the constructor limits validate through the same
`finite_number_error`, `positive_finite_number_error` and
`positive_whole_number_error` domains the three differential-drive bridges
call, so the fourth transport refuses an unusable velocity, hold or count with
byte-identical text rather than a hand-rolled copy of the contract.

`__repr__` renders a half-built instance rather than raising. The constructor
validates `node_name` before assigning it, so a refused construction left an
instance whose `repr` raised `AttributeError: 'AckermannRosRobot' object has no
attribute 'node_name'` - sending a reader after the attribute instead of the
`ValueError: invalid node_name` they already caused. It now delegates to the
single owner of that wording, `strands_robots.utils.partial_construction_repr`,
as the three differential-drive bridges do, and the class is triaged into the
survey in `tests/test_repr_survives_partial_construction.py` (both the
half-built sweep and the documented-refusal cases).

A timed `drive()` whose trailing halt fails now reports that failure instead of
the main publish's success. `use_ros` reports a transport failure as an error
dict rather than raising, and the halt's verdict was discarded - so a node dying
mid-hold left the call returning `success` for a car still holding the commanded
throttle, on a tool whose description promises that a command with a duration
stops itself. The result now names the published command, the failed halt and
its cause, so the agent's next action is `stop`.
