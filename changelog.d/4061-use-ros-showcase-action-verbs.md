### Fixed: the `use_ros` showcase drives the action-client verbs it claims

`examples/ros2/use_ros/showcase.py` is sold by four surfaces as exercising every
`use_ros` action and never called `list_actions` or `action_send_goal` - the two
verbs whose reply is a terminal status and whose expiring timeout cancels the
goal rather than abandoning it - so its captured `sample_output.txt` showed
neither. It now lists the action servers on the graph and sends a goal to
turtlesim's `/turtle1/rotate_absolute`, and its exit code grades the goal's
terminal status together with the heading read back from `/turtle1/pose`.
